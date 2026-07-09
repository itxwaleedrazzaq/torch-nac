import torch
import torch.nn.functional as F

from .neuronal_attention_circuit import NAC, _tf_gather_nd, _pt_activation


class OUWrap(torch.nn.Module):
    """
    Neuronal Stochastic Attention Circuit (NSAC) Wrapper — PyTorch port.

    Wraps NAC dynamics in Ornstein-Uhlenbeck (OU) dynamics, modelling the
    attention process as a stochastic differential equation and providing both
    a predictive mean and a standard deviation (uncertainty).
    """

    def __init__(
        self,
        nac_instance: NAC,
        output_dim: int = 1,
        bn_mean: float = 0.0,
        bn_std: float = 0.1,
        activation=None,
        return_attention: bool = False,
        return_sequences: bool = False,
        return_cell_state: bool = False,
        **kwargs,
    ):
        """
        Args:
            nac_instance: Instance of the (PyTorch) Neuronal Attention Circuit to wrap.
            output_dim: Dimension of the final probabilistic output.
            bn_mean: Mean for the Brownian noise in the OU process.
            bn_std: Standard deviation for the Brownian noise in the OU process.
            activation: Activation function for the hidden state.
            return_attention: Whether to return the attention weight matrix.
            return_sequences: Whether to return the full sequence or the last hidden state.
            return_cell_state: Whether to return the internal cell state of the NSAC.
        """
        super().__init__()

        if not isinstance(nac_instance, NAC):
            raise ValueError("nac_instance must be an instance of NAC")

        self.nac = nac_instance
        self.output_dim = int(output_dim)
        self.bn_mean = float(bn_mean)
        self.bn_std = float(bn_std)
        self.activation = _pt_activation(activation)
        self.return_attention = bool(return_attention)
        self.return_sequences = bool(return_sequences)
        self.return_cell_state = bool(return_cell_state)

        # NCP Projection to generate OU parameters: mu, theta, and sigma.
        # inter→motor projection with output=3 (phi, kappa, psi).
        self.q_proj = self.nac.q_proj
        self.k_proj = self.nac.k_proj
        self.v_proj = self.nac.v_proj
        self.ncp_out = self.nac._make_inter_to_motor_projections("ou_out", output=3)

        # Standard attention output projection.
        self.attention_out_proj = torch.nn.Linear(
            self.nac.d_model, self.nac.d_model * 2, bias=self.nac.use_bias
        )

        # Probabilistic regression output heads (NEW Denses, owned by OUWrap).
        self.mean_head = torch.nn.Linear(self.nac.d_model, self.output_dim, bias=self.nac.use_bias)
        self.std_head = torch.nn.Linear(self.nac.d_model, self.output_dim, bias=self.nac.use_bias)

        # Initialize the OUWrap-owned Dense layers
        self._reset_linear_parameters()

    def _reset_linear_parameters(self):
        """Initialize Dense-equivalent layers to Glorot kernel, zero bias."""
        for layer in (self.attention_out_proj, self.mean_head, self.std_head):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    @staticmethod
    def _resolve_training(explicit, module: torch.nn.Module) -> bool:
        if explicit is not None:
            return bool(explicit)
        return bool(module.training)

    def compute_phi_kappa_psi_time(self, q, k, t):
        """
        Computes the SDE parameters and time interpolation.

        Returns:
            phi: Equilibrium mean [B, H, Tq, K_eff]
            kappa: Mean reversion speed [B, H, Tq, K_eff]
            psi: Volatility (diffusion) [B, H, Tq, K_eff]
            t_interp: Interpolated time steps [B, H, Tq, K_eff]
            topk_idx: Indices of the selected sparse attention keys [B, H, Tq, K_eff]
        """
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        seq_len_q = q.shape[2]

        # Extract sparse top-k pairwise interactions (uses nac's method).
        pair_features, topk_idx = self.nac.sparse_topk_pairwise(q, k, K=self.nac.topk)
        effective_k = pair_features.shape[3]

        # Generate OU parameters from the NCPCell (inter→motor, output=3).
        flat_pairs = pair_features.reshape(-1, pair_features.shape[-1])   # [N, 2D]
        raw_params = self.ncp_out(flat_pairs.unsqueeze(1))                # [N, 1, 3]
        raw_params = raw_params.squeeze(1)                                # [N, 3]

        # Reshape to parameter dimensions [B, H, Tq, K_eff, 3]
        param_tensor = raw_params.reshape(batch_size, num_heads, seq_len_q, effective_k, 3)

        phi = torch.tanh(param_tensor[..., 0])
        kappa = F.softplus(param_tensor[..., 1]) + self.nac.tau_epsilon
        psi = F.softplus(param_tensor[..., 2]) + self.nac.tau_epsilon

        # Compute time-dependent gates.
        tab = self.nac.time_ab(pair_features)      # [B, H, Tq, K_eff, 2]
        time_gate_a = tab[..., :1]                  # [B, H, Tq, K_eff, 1]
        time_gate_b = tab[..., 1:]                  # [B, H, Tq, K_eff, 1]

        # Handle time scalar or tensor logic.
        if t is None:
            t_val = torch.ones((), dtype=pair_features.dtype, device=pair_features.device)
            t_expanded = t_val.reshape(1, 1, 1, 1, 1)
        else:
            t_val = t.to(dtype=pair_features.dtype)
            t_rank = t_val.ndim
            if t_rank == 0:                         # scalar
                t_expanded = t_val.reshape(1, 1, 1, 1, 1)
            elif t_rank == 1:                       # [Tq]
                t_expanded = t_val.reshape(1, 1, -1, 1, 1)
            else:                                   # [B, Tq] or broadcastable
                t_expanded = t_val.reshape(batch_size, 1, seq_len_q, 1, 1)
            t_expanded = t_expanded.expand(batch_size, 1, seq_len_q, 1, 1)

        t_interp = torch.sigmoid(-time_gate_a * t_expanded + time_gate_b)[..., 0]

        return phi, kappa, psi, t_interp, topk_idx

    def forward(self, inputs, mask=None, training=None):
        """
        Forward computation solving the OU dynamics for stochastic attention.
        """
        # ---- Unpack mask ----
        if isinstance(mask, (list, tuple)):
            mask = mask[0] if len(mask) > 0 else None
            if mask is None or (isinstance(mask, torch.Tensor) and mask.numel() == 0):
                mask = None

        # ---- Unpack inputs ----
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                x, t = inputs
                q_in = k_in = v_in = x
            elif len(inputs) == 3:
                q_in, k_in, v_in = inputs
                t = None
            elif len(inputs) == 4:
                q_in, k_in, v_in, t = inputs
            else:
                raise ValueError("Unsupported input tuple length")
        else:
            q_in = k_in = v_in = inputs
            t = None

        # Resolve the effective training flag for dropout / RNG-dependent paths.
        training_eff = self._resolve_training(training, self)

        # Project through NAC's Sensory Gate (shared submodules).
        q = self.nac.q_proj(q_in)
        k = self.nac.k_proj(k_in)
        v = self.nac.v_proj(v_in)

        # Multi-Head NAC Splitting.
        qh = self.nac.split_heads(q)
        kh = self.nac.split_heads(k)
        vh = self.nac.split_heads(v)

        B, H, Tq, _ = qh.shape

        # Compute OU Parameters.
        phi, kappa, psi, dt, topk_idx = self.compute_phi_kappa_psi_time(qh, kh, t)

        # OU Mean and Variance.
        ou_mean = phi * (1.0 - torch.exp(-kappa * dt))
        ou_var = (psi ** 2) * (1.0 - torch.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
        ou_stddev = torch.sqrt(torch.clamp(ou_var, min=1e-9))

        # Brownian Motion Realization.
        # PyTorch: torch.randn(shape)*stddev + mean. Identical in distribution.
        noise = ou_stddev * (
            torch.randn(ou_stddev.shape, dtype=ou_stddev.dtype,
                        device=ou_stddev.device) * self.bn_std + self.bn_mean
        )

        attn_logits = ou_mean + noise

        # Apply Sparse Masking.
        if mask is not None:
            mask_f = mask.to(attn_logits.dtype)                  # [B, Tk]
            mask_b = mask_f[:, None, None, :]                     # [B, 1, 1, Tk]
            mask_b = mask_b.expand(B, H, Tq, mask.shape[-1])      # [B, H, Tq, Tk]
            mask_gathered = _tf_gather_nd(mask_b, topk_idx,
                                          batch_dims=3, axis=3)   # [B, H, Tq, K_eff]
            attn_logits = attn_logits * mask_gathered

        # logistic-normalization distribution.
        attn_weights = F.softmax(attn_logits, dim=-1)
        # Use functional dropout honoring the explicit training flag.
        attn_weights = F.dropout(attn_weights, p=self.nac.dropout_rate,
                                 training=training_eff)

        # Head-specific output computation.
        vh_topk = _tf_gather_nd(vh, topk_idx, batch_dims=2, axis=2)   # [B,H,Tq,K_eff,depth]
        weighted = attn_weights.unsqueeze(-1) * vh_topk               # [B,H,Tq,K_eff,depth]
        head_out = torch.sum(weighted, dim=3)                         # [B,H,Tq,depth]

        # Multihead extension with double-dimension projection.
        combined = self.nac.combine_heads(head_out)                   # [B, Tq, d_model]
        projected = self.attention_out_proj(combined)                 # [B, Tq, d_model*2]

        if self.activation is not None:
            projected = self.activation(projected)

        # Split into predictive and uncertainty streams.
        features_mean, features_std = torch.chunk(projected, 2, dim=-1)

        final_mean = self.mean_head(features_mean)   # [B, Tq, output_dim]
        final_std = self.std_head(features_std)      # [B, Tq, output_dim]

        # Optionally return only the last time step.
        if not self.return_sequences:
            final_mean = final_mean[:, -1, :]
            final_std = final_std[:, -1, :]

        # Optionally extract cell voltages.
        cell_voltages = None
        if self.return_cell_state:
            cell_voltages = self.cell_state(q_in, k_in, v_in)

        # Return values based on flags.
        if self.return_attention and cell_voltages is not None:
            return final_mean, final_std, attn_weights, cell_voltages
        elif self.return_attention:
            return final_mean, final_std, attn_weights
        elif cell_voltages is not None:
            return final_mean, final_std, cell_voltages

        return final_mean, final_std

    # cell_state: membrane voltages for ncp_out + sensory projections
    def cell_state(self, q_in, k_in, v_in):
        """
        Compute cell_state (membrane voltages).
        """
        mmvs = {}

        q_state = None
        k_state = None
        v_state = None
        out_state = [None] * self.nac.num_heads

        qh = self.nac.split_heads(self.nac.q_proj(q_in))
        kh = self.nac.split_heads(self.nac.k_proj(k_in))

        head_mmvs = []

        for h in range(self.nac.num_heads):
            pair, _ = self.nac.sparse_topk_pairwise(
                qh[:, h:h + 1, :, :],
                kh[:, h:h + 1, :, :],
                K=self.nac.topk,
            )

            B = pair.shape[0]
            Tq = pair.shape[2]
            K_eff = pair.shape[3]
            D2 = pair.shape[4]

            sensory_size = int(self.ncp_out.cell.sensory_size)

            # Conditionally truncate or pad the last dimension.
            if D2 > sensory_size:
                pair = pair[..., :sensory_size]
            else:
                pad_size = sensory_size - D2
                pair = F.pad(pair, (0, pad_size), mode='constant', value=0.0)

            pair_combined = torch.mean(
                pair.reshape(B, Tq, K_eff, sensory_size),
                dim=2,
            )   # [B, Tq, sensory_size]

            mmv, out_state[h] = self.nac.extract_membrane_voltages(
                self.ncp_out,
                pair_combined,
                state=out_state[h],
            )
            head_mmvs.append(mmv)

        mmvs["ncp_out"] = torch.cat(head_mmvs, dim=-1)

        mmvs["q_proj"], q_state = self.nac.extract_membrane_voltages(self.nac.q_proj, q_in, state=q_state)
        mmvs["k_proj"], k_state = self.nac.extract_membrane_voltages(self.nac.k_proj, k_in, state=k_state)
        mmvs["v_proj"], v_state = self.nac.extract_membrane_voltages(self.nac.v_proj, v_in, state=v_state)

        return mmvs

    # ------------------------------------------------------------------
    # Extra representation
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"output_dim={self.output_dim}, bn_mean={self.bn_mean}, "
            f"bn_std={self.bn_std}, return_attention={self.return_attention}, "
            f"return_sequences={self.return_sequences}, "
            f"return_cell_state={self.return_cell_state}"
        )
