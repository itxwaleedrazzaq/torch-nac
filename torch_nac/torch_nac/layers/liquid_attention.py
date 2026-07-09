import torch
import torch.nn.functional as F


def _tf_gather_nd(params: torch.Tensor, indices: torch.Tensor,
                  batch_dims: int, axis: int) -> torch.Tensor:
    full_idx = []
    for d in range(params.ndim):
        if d < axis:
            if d < batch_dims:
                # Batch dimension: create broadcast arange matching indices shape
                shape = [1] * indices.ndim
                shape[d] = params.shape[d]
                full_idx.append(
                    torch.arange(params.shape[d], device=indices.device,
                                 dtype=torch.long)
                    .reshape(shape).expand_as(indices)
                )
            else:
                full_idx.append(slice(None))
        elif d == axis:
            full_idx.append(indices)
        else:
            full_idx.append(slice(None))
    return params[tuple(full_idx)]


# Activation helper
def _pt_activation(activation):
    if activation is None:
        return torch.nn.Identity()
    if isinstance(activation, str):
        name = activation.lower()
        if name == "linear":
            return torch.nn.Identity()
        if name == "relu":
            return F.relu
        if name == "sigmoid":
            return torch.sigmoid
        if name == "tanh":
            return torch.tanh
        if name == "softplus":
            return F.softplus
        if name == "elu":
            return F.elu
        if name == "selu":
            return F.selu
        if name == "gelu":
            return F.gelu
        if name in ("swish", "silu"):
            return F.silu
        raise ValueError(f"Unknown activation name: {activation}")
    # Fallback: assume it's a callable.
    # Map by __name__ if available; otherwise return as-is (torch-compatible).
    if callable(activation):
        try:
            name = activation.__name__.lower() if hasattr(activation, '__name__') else ""
            if "relu" in name:
                return F.relu
            if "sigmoid" in name:
                return torch.sigmoid
            if "tanh" in name:
                return torch.tanh
            if "softplus" in name:
                return F.softplus
            if "elu" in name:
                return F.elu
            if "selu" in name:
                return F.selu
            if "gelu" in name:
                return F.gelu
        except Exception:
            pass
        return activation
    return torch.nn.Identity()


# LAN: Liquid Attention Network
class LAN(torch.nn.Module):
    """
    Liquid Attention Network (LAN) — PyTorch port.

    Implements attention with:
      - LSTM-based gating for computing phi (content gate) and tau (time constant)
      - Euler-integrated softmax logits
      - Optional pairwise (dense) or sparse top-k attention
      - Optional sink gate modulation
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        topk: int = 8,
        euler_steps: int = 2,
        use_sink_gate: bool = True,
        activation=None,
        tau_epsilon: float = 1e-6,
        delta_t: float = 0.01,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_pairwise: bool = False,
        return_attention: bool = False,
        return_sequences: bool = False,
        jit_compile: bool = True,  # kept for API compatibility; no-op in PyTorch
        input_dim: int = None,     # feature dim of q/k/v inputs; defaults to d_model
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert int(euler_steps) >= 1, "euler_steps must be >= 1"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.topk = int(topk)
        self.euler_steps = int(euler_steps)
        self.delta_t = float(delta_t)
        self.use_sink_gate = use_sink_gate
        self.tau_epsilon = tau_epsilon
        self.dropout_rate = dropout
        self.use_bias = use_bias
        self.use_pairwise = use_pairwise
        self.return_attention = return_attention
        self.return_sequences = return_sequences
        self.jit_compile = jit_compile  # stored for get_config parity
        self.input_dim = int(input_dim) if input_dim is not None else self.d_model

        # ---- Activation ----
        self.activation = _pt_activation(activation)

        # ---- Linear projections ----
        self.q_dense = torch.nn.Linear(self.input_dim, self.d_model, bias=self.use_bias)
        self.k_dense = torch.nn.Linear(self.input_dim, self.d_model, bias=self.use_bias)
        self.v_dense = torch.nn.Linear(self.input_dim, self.d_model, bias=self.use_bias)
        self.out_dense = torch.nn.Linear(self.d_model, self.d_model, bias=self.use_bias)

        # ---- Recurrent gating ----
        self.gate_in = torch.nn.LSTM(
            input_size=2 * self.depth,
            hidden_size=self.d_model,
            bias=self.use_bias,
            batch_first=True,
        )
        self.gate_out = torch.nn.Linear(self.d_model, 1, bias=self.use_bias)

        # ---- Sink gate ----
        # Input width is input_dim (matches q/k/v), output is d_model.
        self.sink_gate = torch.nn.Linear(self.input_dim, self.d_model, bias=True)

        # ---- Dropout ----
        self.attn_dropout = torch.nn.Dropout(self.dropout_rate)

        # ---- Initialize weights matching TF defaults ----
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights to match TF/Keras defaults."""
        for module in [self.q_dense, self.k_dense, self.v_dense, self.out_dense,
                       self.gate_out, self.sink_gate]:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # zeros for bias. Override PyTorch defaults to match.
        for name, param in self.gate_in.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

    # Multi-head utilities
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, T, d_model] → [B, num_heads, T, depth]
        """
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # [B, num_heads, T, depth]

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, num_heads, T, depth] → [B, T, d_model]
        """
        x = x.permute(0, 2, 1, 3)  # [B, T, num_heads, depth]
        B, T, _, _ = x.shape
        return x.reshape(B, T, self.d_model)

    # Pairwise concatenation (dense mode)
    def pairwise_concat(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Concatenate each query with each key across sequence positions.

        q: [B, H, Tq, D]
        k: [B, H, Tk, D]
        Returns: [B, H, Tq, Tk, 2D]
        """
        q_exp = q.unsqueeze(3)          # [B, H, Tq, 1, D]
        k_exp = k.unsqueeze(2)          # [B, H, 1, Tk, D]
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        q_bc = q_exp.expand(B, H, Tq, Tk, D)
        k_bc = k_exp.expand(B, H, Tq, Tk, D)
        return torch.cat([q_bc, k_bc], dim=-1)

    # Sparse top-k selection (exact global top-k)
    def sparse_topk_pairwise(self, q: torch.Tensor, k: torch.Tensor,
                             K: int = None):
        """
        Compute pairwise dot products and select top-K keys per query.

        This is EXACT global top-k (unlike NAC's approximate square-root
        partitioning).

        q: [B, H, Tq, D]
        k: [B, H, Tk, D]
        K: override top-k value

        Returns:
          topk_pairs: [B, H, Tq, K_eff, 2D]
          topk_idx:   [B, H, Tq, K_eff]
        """
        if K is None:
            K = self.topk
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k)  # [B, H, Tq, Tk]
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        K_eff = min(int(K), Tk)
        _, topk_idx = torch.topk(scores, k=K_eff, dim=-1)  # [B, H, Tq, K_eff]

        # Gather selected keys
        k_gathered = _tf_gather_nd(k, topk_idx, batch_dims=2, axis=2)
        # [B, H, Tq, K_eff, D]

        # Broadcast q to match selected keys
        q_exp = q.unsqueeze(3)                          # [B, H, Tq, 1, D]
        q_bc = q_exp.expand(B, H, Tq, K_eff, D)
        topk_pairs = torch.cat([q_bc, k_gathered], dim=-1)
        return topk_pairs, topk_idx

    # Compute phi (gate) and tau (time constant)
    def compute_phi_tau(self, q: torch.Tensor, k: torch.Tensor):
        """
        Compute phi (target-content gate) and tau (time constant gate).

        q, k: [B, H, T, D]

        Returns:
          phi: [B, H, Tq, Tk]   (pairwise mode) or [B, H, Tq, K_eff] (sparse mode)
          tau: [B, H, Tq, Tk]   (pairwise mode) or [B, H, Tq, K_eff] (sparse mode)
          idx: top-k indices (None in pairwise mode)
        """
        if self.use_pairwise:
            pair = self.pairwise_concat(q, k)
            idx = None
        else:
            pair, idx = self.sparse_topk_pairwise(q, k)

        B, H, Tq, Tk, D2 = pair.shape

        # Reshape for LSTM: [B*H*Tq, Tk, 2D]
        pair = pair.reshape(B * H * Tq, Tk, D2)

        # LSTM gating over the key dimension
        x, _ = self.gate_in(pair)            # [B*H*Tq, Tk, d_model]
        gate_raw = self.gate_out(x)          # [B*H*Tq, Tk, 1]
        gate_raw = gate_raw.reshape(B, H, Tq, Tk)  # [B, H, Tq, Tk]

        phi = F.relu(gate_raw)                        # Compute phi
        tau = F.softplus(gate_raw) + self.tau_epsilon # Compute tau

        return phi, tau, idx

    # Euler softmax core
    def _euler_softmax_core(self, phi: torch.Tensor, tau: torch.Tensor,
                            mask_bc: torch.Tensor = None) -> torch.Tensor:
        """
        Euler-integrated softmax with optional masking.

        mask_bc is pre-shaped to broadcast against attn_logits:
          - [B, 1, 1, Tk] in pairwise mode
          - [B, H, Tq, K_eff] in sparse mode
          - None if no mask
        """
        dt = torch.tensor(self.delta_t, dtype=phi.dtype, device=phi.device)

        # Euler stability condition: dt <= 1 / sup(tau)
        tau_max = tau.max()
        dt_max = 1.0 / (tau_max + 1e-12)
        dt = torch.min(dt, dt_max)

        # Euler integration for attention logits
        a = torch.zeros_like(phi)
        for _ in range(self.euler_steps):
            increment = dt * (-tau * a + phi)
            a = a + increment
        attn_logits = a

        # Apply mask via torch.where (see docstring)
        if mask_bc is not None:
            mask_bool = mask_bc.to(torch.bool)
            very_neg = torch.tensor(-1e4, dtype=attn_logits.dtype,
                                    device=attn_logits.device)
            attn_logits = torch.where(mask_bool, attn_logits, very_neg)

        attn_weights = F.softmax(attn_logits, dim=-1)
        return attn_weights

    # Forward pass
    def forward(self, inputs, mask=None):
        """
        Forward pass for LAN.

        Args:
          inputs: can be
            - single tensor (self-attention)  [B, T, d_model]
            - tuple of 2 tensors ((q, k), v)
            - tuple of 3 tensors (q, k, v)
          mask: optional attention mask [B, Tk] (key-stream padding mask,
                1=valid, 0=masked)

        Returns:
          Attention output (and optionally weights)
        """
        # ---- Unpack mask ----
        if isinstance(mask, (list, tuple)):
            mask = mask[0] if len(mask) > 0 else None
            if mask is None or (isinstance(mask, torch.Tensor) and mask.numel() == 0):
                mask = None

        # ---- Unpack inputs ----
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            x, v_in = inputs
            q_in = k_in = x
        elif isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            q_in, k_in, v_in = inputs
        else:
            q_in = k_in = v_in = inputs

        # ---- Linear projections ----
        q = self.q_dense(q_in)   # [B, T, d_model]
        k = self.k_dense(k_in)   # [B, T, d_model]
        v = self.v_dense(v_in)   # [B, T, d_model]

        # ---- Split heads ----
        qh = self.split_heads(q)  # [B, H, T, depth]
        kh = self.split_heads(k)
        vh = self.split_heads(v)

        # ---- Compute phi, tau ----
        phi, tau, topk_idx = self.compute_phi_tau(qh, kh)

        # ---- Prepare mask ----
        if mask is None:
            mask_bc = None
        elif self.use_pairwise:
            mask_bc = mask.to(phi.dtype)
            mask_bc = mask_bc.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, Tk]
        else:
            mask_idx = mask.to(topk_idx.dtype)
            mask_bc = _tf_gather_nd(mask_idx, topk_idx, batch_dims=1, axis=1)
            mask_bc = mask_bc.to(phi.dtype)

        # ---- Euler softmax ----
        attn_weights = self._euler_softmax_core(phi, tau, mask_bc)
        attn_weights = self.attn_dropout(attn_weights)

        # ---- Integrated output ----
        if self.use_pairwise:
            # Dense attention: [B, H, Tq, Tk] @ [B, H, Tk, D] → [B, H, Tq, D]
            output = torch.matmul(attn_weights, vh)
        else:
            # Sparse attention: gather values from top-k indices
            vh_topk = _tf_gather_nd(vh, topk_idx, batch_dims=2, axis=2)
            # [B, H, Tq, K_eff, D]
            output = torch.einsum("bhqk,bhqkd->bhqd", attn_weights, vh_topk)

        # ---- Combine heads and final projection ----
        combined = self.combine_heads(output)  # [B, T, d_model]

        # Sink gate modulation
        if self.use_sink_gate:
            sink_gate_values = torch.sigmoid(self.sink_gate(q_in))
            combined = combined * sink_gate_values

        out = self.activation(self.out_dense(combined))

        # ---- Return options ----
        result = out if self.return_sequences else out[:, -1, :]
        if self.return_attention:
            return result, attn_weights
        return result

    # Extra API for compatibility
    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, num_heads={self.num_heads}, "
                f"topk={self.topk}, euler_steps={self.euler_steps}, "
                f"delta_t={self.delta_t}, use_sink_gate={self.use_sink_gate}, "
                f"use_pairwise={self.use_pairwise}, "
                f"return_attention={self.return_attention}, "
                f"return_sequences={self.return_sequences}")