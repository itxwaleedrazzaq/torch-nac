import math
import torch
import numpy as np
import torch.nn.functional as F
from ncps.wirings import AutoNCP


# Helper function
def _tf_gather_nd(params: torch.Tensor, indices: torch.Tensor,
                  batch_dims: int, axis: int) -> torch.Tensor:
    """
    Uses PyTorch advanced indexing to replicate the TF semantics exactly.

    Args:
        params:      source tensor
        indices:     integer index tensor (shares leading batch_dims with params)
        batch_dims:  number of shared leading batch dimensions
        axis:        axis of params to gather along (0-indexed)

    Returns:
        Gathered tensor with shape:
          params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis+1:]
    """
    full_idx = []
    for d in range(params.ndim):
        if d < axis:
            if d < batch_dims:
                # Batch axis: arange broadcast to match indices shape
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
    """
    Accepts: None, 'linear', 'relu', 'sigmoid', 'tanh', 'softplus',
    'elu', 'selu', 'gelu', 'swish'/'silu', or a torch-compatible callable.
    """
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
    if callable(activation):
        # Check if it's already torch-compatible (e.g., torch.relu)
        try:
            name = getattr(activation, '__name__', '').lower()
            if 'relu' in name:
                return F.relu
            if 'sigmoid' in name:
                return torch.sigmoid
            if 'tanh' in name:
                return torch.tanh
            if 'softplus' in name:
                return F.softplus
            if 'elu' in name:
                return F.elu
            if 'selu' in name:
                return F.selu
            if 'gelu' in name:
                return F.gelu
        except Exception:
            pass
        return activation
    return torch.nn.Identity()


# NCPCell: Repurposed Neuronal Circuit Policy recurrent cell
class NCPCell(torch.nn.Module):
    """
    Neuronal Circuit Policy (NCP) recurrent cell — PyTorch port.

    Implements a recurrent neural network cell based on worm-brain inspired
    connectivity patterns. Each call processes one time step.

    Used as a step function inside _NCPWrapper for sequence processing.
    """

    def __init__(
        self,
        wiring: AutoNCP,
        input_dim: int,
        activation: str = "linear",
        input_group: str = "sensory",
        output_group: str = "motor",
        disabled_groups: list = None,
    ):
        super().__init__()
        self._wiring = wiring
        self._activation_name = activation
        self._input_group = input_group
        self._output_group = output_group
        self._disabled_groups = disabled_groups if disabled_groups is not None else []

        self.state_size = 0
        self.sensory_size = 0
        self.motor_size = 0
        self.output_size = 0
        self.input_indices = None
        self.output_indices = None
        self._built = False

        if activation == "linear" or activation is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = _pt_activation(activation)

        # Build immediately instead of waiting for first forward()
        self.build(input_dim)

    @property
    def wiring(self):
        return self._wiring

    def build(self, input_dim: int):
        """
        Create trainable parameters and neuron group indices based on the wiring.

        Args:
            input_dim: Feature dimension of the inputs (int).

        """
        if self._built:
            return

        # Build the wiring for the given input size
        self._wiring.build(input_dim)

        # ---- Identify neuron groups (exact TF logic) ----
        sensory_adj = np.abs(self._wiring.sensory_adjacency_matrix)
        self._sensory_indices = np.where(np.sum(sensory_adj, axis=0) > 0)[0]
        self._motor_indices = np.arange(self._wiring.output_dim)

        # Derive inter and command neuron indices
        command_full = np.setdiff1d(
            np.arange(self._wiring.units),
            np.union1d(self._sensory_indices, self._motor_indices)
        )
        if len(command_full) > 0:
            cmd_adj = np.abs(self._wiring.adjacency_matrix)[
                command_full[:, None], command_full
            ]
            incoming = np.sum(cmd_adj, axis=0) > 0
            self._inter_indices = command_full[~incoming]
            self._command_indices = command_full[incoming]
        else:
            self._inter_indices = np.array([], dtype=int)
            self._command_indices = np.array([], dtype=int)

        # ---- Disable neuron groups as requested ----
        disabled_indices = np.array([], dtype=int)
        for group in self._disabled_groups:
            if group == 'sensory':
                disabled_indices = np.union1d(disabled_indices, self._sensory_indices)
            elif group == 'inter':
                disabled_indices = np.union1d(disabled_indices, self._inter_indices)
            elif group == 'command':
                disabled_indices = np.union1d(disabled_indices, self._command_indices)
            elif group == 'motor':
                disabled_indices = np.union1d(disabled_indices, self._motor_indices)
            else:
                raise ValueError(f"Unknown group to disable: {group}")

        # ---- active_mask ----
        active_mask_value = np.ones((self._wiring.units,), dtype="float32")
        active_mask_value[disabled_indices] = 0.0
        self.register_buffer("active_mask",
                             torch.from_numpy(active_mask_value))

        # ---- Group map ----
        group_map = {
            'sensory': self._sensory_indices,
            'inter': self._inter_indices,
            'command': self._command_indices,
            'motor': self._motor_indices,
            'all': np.arange(self._wiring.units)
        }

        if self._input_group not in group_map:
            raise ValueError(f"Unknown input_group: {self._input_group}")
        if self._output_group not in group_map:
            raise ValueError(f"Unknown output_group: {self._output_group}")

        self.input_indices = group_map[self._input_group]
        self.output_indices = group_map[self._output_group]

        # ---- Store sizes ----
        self.state_size = self._wiring.units
        self.sensory_size = self._wiring.input_dim  # set by wiring.build()
        self.motor_size = self._wiring.output_dim
        self.output_size = len(self.output_indices) or self.motor_size

        # ---- Trainable parameters ----
        # input_kernel: [sensory_size, state_size]
        self.input_kernel = torch.nn.Parameter(
            torch.empty(self.sensory_size, self.state_size)
        )
        # recurrent_kernel: [state_size, state_size]
        self.recurrent_kernel = torch.nn.Parameter(
            torch.empty(self.state_size, self.state_size)
        )
        # bias: [state_size]
        self.bias = torch.nn.Parameter(torch.empty(self.state_size))

        # ---- Fixed sparse connectivity masks (non-trainable buffers) ----
        sparsity_mask_value = np.abs(self._wiring.adjacency_matrix).astype("float32")
        self.register_buffer("sparsity_mask",
                             torch.from_numpy(sparsity_mask_value))

        sensory_mask = np.zeros((self.sensory_size, self.state_size), dtype="float32")
        sensory_mask[:, self.input_indices] = 1.0
        self.register_buffer("sensory_sparsity_mask",
                             torch.from_numpy(sensory_mask))

        # ---- Input / output affine transforms ----
        self.input_w = torch.nn.Parameter(torch.ones(self.sensory_size))
        self.input_b = torch.nn.Parameter(torch.zeros(self.sensory_size))
        self.output_w = torch.nn.Parameter(torch.ones(len(self.output_indices)))
        self.output_b = torch.nn.Parameter(torch.zeros(len(self.output_indices)))

        # ---- Initialize weights----
        self._reset_parameters()
        self._built = True

    def _reset_parameters(self):
        # input_kernel: GlorotUniform (Xavier uniform)
        torch.nn.init.xavier_uniform_(self.input_kernel)
        # recurrent_kernel: Orthogonal
        torch.nn.init.orthogonal_(self.recurrent_kernel)
        # bias: Zeros
        torch.nn.init.zeros_(self.bias)

    @property
    def masked_input_kernel(self) -> torch.Tensor:
        """input_kernel * sensory_sparsity_mask, recomputed per access."""
        return self.input_kernel * self.sensory_sparsity_mask

    @property
    def masked_recurrent_kernel(self) -> torch.Tensor:
        """recurrent_kernel * sparsity_mask, recomputed per access."""
        return self.recurrent_kernel * self.sparsity_mask

    # -- Input / output affine maps --
    def _map_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Affine transform: inputs * input_w + input_b"""
        return inputs * self.input_w + self.input_b

    def _map_outputs(self, state: torch.Tensor) -> torch.Tensor:
        """Select output neurons and apply affine scaling."""
        output = state[:, self.output_indices]
        return output * self.output_w + self.output_b

    # -- Single-step forward --
    def forward(self, inputs: torch.Tensor, state: torch.Tensor):
        """
        Perform one recurrent step.

        Args:
            inputs: Current sensory input [B, sensory_size]
            state:  Previous neuron state [B, state_size]

        Returns:
            outputs:    [B, output_size]
            next_state: [B, state_size]
        """
        inputs = self._map_inputs(inputs)

        # Recurrent + sensory contributions with sparse connectivity masks
        recurrent = torch.matmul(state, self.masked_recurrent_kernel)
        sensory = torch.matmul(inputs, self.masked_input_kernel)

        # State update: activation(…) * active_mask
        next_state = self.activation(recurrent + sensory + self.bias)
        next_state = next_state * self.active_mask

        outputs = self._map_outputs(next_state)
        return outputs, next_state

    def init_state(self, batch_size: int, dtype=None, device=None) -> torch.Tensor:
        """Zero initial state [batch_size, state_size]."""
        if dtype is None:
            dtype = torch.float32
        return torch.zeros(batch_size, self.state_size, dtype=dtype, device=device)


# _NCPWrapper: unroll an NCPCell over a sequence dimension
class _NCPWrapper(torch.nn.Module):
    """
    Unrolls the NCPCell over the time dimension (dim=1 with batch_first=True)
    and returns all output timesteps.

    The cell is lazily built on the first forward pass (PyTorch convention).
    """

    def __init__(self, cell: NCPCell):
        super().__init__()
        self.cell = cell
        # Use a custom attribute (not .name which clashes with torch.nn.Module internals)
        self._wrapper_name = ""

    @property
    def name(self):
        return self._wrapper_name

    @name.setter
    def name(self, val: str):
        self._wrapper_name = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, input_dim = x.shape
        state = self.cell.init_state(B, dtype=x.dtype, device=x.device)
        outputs = []
        for t in range(T):
            out_t, state = self.cell(x[:, t, :], state)
            outputs.append(out_t)
        return torch.stack(outputs, dim=1)


# NAC: Neuronal Attention Circuit
class NAC(torch.nn.Module):
    """
    Neuronal Attention Circuit (NAC) — PyTorch port.

    A CT-Attention mechanism using Neuronal Circuit Policies (NCPs).
    Sparsifies pairwise concatenations to the top-k elements. Queries, keys,
    and values are projected through NCP-based sensory neurons, and phi and
    tau are computed via inter → command → motor pathways.

    Uses AutoNCP from ncps.wirings as internal
    mechanism for constructing NCP-based neurons.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        topk: int = 8,
        mode: str = 'exact',            # 'steady', 'euler', or 'exact'
        euler_steps: int = 5,
        sparsity: float = 0.5,
        delta_t: float = 0.5,
        activation=None,
        dropout: float = 0.0,
        tau_epsilon: float = 1e-6,
        use_bias: bool = True,
        use_riemann_sum: bool = True,
        return_attention: bool = False,
        return_sequences: bool = False,
        return_cell_state: bool = False,
        input_dim: int = None,

    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert 0.1 <= sparsity <= 0.9, "sparsity must be in [0.1, 0.9]"
        assert mode in ('steady', 'euler', 'exact'), \
            "mode must be 'steady', 'euler' or 'exact'"

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.depth = self.d_model // self.num_heads
        self.topk = int(topk)
        self.mode = mode
        self.euler_steps = int(euler_steps)
        self.sparsity = float(sparsity)
        self.delta_t = float(delta_t)
        self.tau_epsilon = float(tau_epsilon)
        self.dropout_rate = float(dropout)
        self.use_bias = bool(use_bias)
        self.use_riemann_sum = bool(use_riemann_sum)
        self.return_attention = bool(return_attention)
        self.return_sequences = bool(return_sequences)
        self.return_cell_state = bool(return_cell_state)
        self.input_dim = int(input_dim) if input_dim is not None else self.d_model

        self.activation = _pt_activation(activation)

        # ---- Projections for q, k, v (sensory NCPs) ----
        self.q_proj = self._make_sensory_projections("q_proj")
        self.k_proj = self._make_sensory_projections("k_proj")
        self.v_proj = self._make_sensory_projections("v_proj")

        # ---- Time MLP (fused Dense(2) replacing two Dense(1)) ----
        # The pair has 2*depth features → mapped to 2 scalars (a, b)
        self.time_ab = torch.nn.Linear(2 * self.depth, 2, bias=True)

        # ---- out_ncp: inter → motor projections for phi/tau ----
        self.out_ncp = self._make_inter_to_motor_projections("out")

        # ---- Dropout ----
        self.attn_dropout = torch.nn.Dropout(self.dropout_rate)

        # ---- Output projection ----
        self.out_dense = torch.nn.Linear(self.d_model, self.d_model, bias=self.use_bias)

        # ---- Initialize non-NCP weights ----
        self._reset_linear_parameters()

    def _reset_linear_parameters(self):
        """Initialize Dense-equivalent layers."""
        # time_ab: Dense(2) — glorot_uniform kernel, zeros bias
        torch.nn.init.xavier_uniform_(self.time_ab.weight)
        torch.nn.init.zeros_(self.time_ab.bias)
        # out_dense: Dense(d_model) — glorot_uniform kernel, zeros bias
        torch.nn.init.xavier_uniform_(self.out_dense.weight)
        if self.out_dense.bias is not None:
            torch.nn.init.zeros_(self.out_dense.bias)

    # Multi-head utilities
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] → [B, H, T, depth]"""
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, T, depth] → [B, T, d_model]"""
        x = x.permute(0, 2, 1, 3)
        B, T, _, _ = x.shape
        return x.reshape(B, T, self.d_model)

    # Projection builders
    def _make_sensory_projections(self, name: str, output: int = 0):
        """
        Create an NCP-based RNN for sensory projections (q, k, v).
        Matches NAC._make_sensory_projections.
        """
        sensory_units = math.ceil((self.d_model - 0.5) / 0.6)
        wiring = AutoNCP(sensory_units, output, sparsity_level=self.sparsity)
        cell = NCPCell(
            wiring,
            input_dim = self.input_dim,  
            activation="linear",
            input_group="sensory",
            output_group="sensory",
            disabled_groups=["inter", "command", "motor"]
        )
        wrapper = _NCPWrapper(cell)
        wrapper.name = name
        return wrapper

    def _make_inter_to_motor_projections(self, name: str, output: int = 1):
        """
        Create an NCP-based RNN mapping inter neurons to motor neurons.
        Matches NAC._make_inter_to_motor_projections in.
        """
        units = self.d_model + int(math.floor(self.d_model / 0.6))
        wiring = AutoNCP(units, output, sparsity_level=self.sparsity)
        cell = NCPCell(
            wiring,
            input_dim=2 * self.depth,   
            activation="linear",
            input_group="inter",
            output_group="motor",
            disabled_groups=["sensory"]
        )
        wrapper = _NCPWrapper(cell)
        wrapper.name = name
        return wrapper

    # Sparse top-k (approximate square-root partitioning)
    def sparse_topk_pairwise(self, q: torch.Tensor, k: torch.Tensor,
                             K: int = None):
        """
        Subquadratic sparse top-k using Square-Root partitioning.

        This algorithm is intentionally approximate (selects best within
        top coarse blocks, not global exact top-k). Do NOT "fix" this to
        be exact — the approximation is part of the design.

        q: [B, H, Tq, D]
        k: [B, H, Tk, D]
        K: number of keys selected per query

        Returns:
          topk_pairs: [B, H, Tq, K_eff, 2D]
          topk_idx:   [B, H, Tq, K_eff]

        Complexity: O(n * sqrt(n) * D) time, O(n * D) space.
        """
        if K is None:
            K = self.topk

        B, H, Tq, D = q.shape
        Tk = k.shape[2]

        # Block size: floor(sqrt(Tk)), at least 1
        block_size = int(math.sqrt(float(Tk)))
        block_size = max(1, block_size)
        num_blocks = Tk // block_size

        Tk_rounded = num_blocks * block_size
        k_cut = k[:, :, :Tk_rounded, :]  # [B, H, Tk_rounded, D]

        # Reshape into blocks, compute centroids
        k_blocks = k_cut.reshape(B, H, num_blocks, block_size, D)
        k_centroids = k_blocks.mean(dim=3)  # [B, H, num_blocks, D]

        # Coarse scores: q · centroid per block
        coarse_scores = torch.einsum("bhqd,bhmd->bhqm", q, k_centroids)
        # [B, H, Tq, num_blocks]

        m_blocks = min(num_blocks, (K // block_size) + 1)
        _, top_block_indices = torch.topk(coarse_scores, k=m_blocks, dim=-1)
        # [B, H, Tq, m_blocks]

        # Gather candidate blocks → flatten
        k_candidates = _tf_gather_nd(k_blocks, top_block_indices,
                                     batch_dims=2, axis=2)
        # [B, H, Tq, m_blocks, block_size, D]
        k_candidates = k_candidates.reshape(B, H, Tq, -1, D)
        # [B, H, Tq, m_blocks * block_size, D]

        # Refined scores
        refined_scores = torch.einsum("bhqd,bhqmd->bhqm", q, k_candidates)

        K_eff = min(K, k_candidates.shape[3])
        _, local_idx = torch.topk(refined_scores, k=K_eff, dim=-1)
        # [B, H, Tq, K_eff]

        # Reconstruct global indices: block_offsets + intra_block_offsets
        block_offsets = top_block_indices.unsqueeze(-1) * block_size
        # [B, H, Tq, m_blocks, 1]
        intra_offsets = torch.arange(block_size, device=k.device, dtype=torch.long)
        intra_offsets = intra_offsets.view(1, 1, 1, 1, block_size)
        global_candidate_map = (block_offsets + intra_offsets)
        # [B, H, Tq, m_blocks, block_size]
        global_candidate_map = global_candidate_map.reshape(B, H, Tq, -1)
        # [B, H, Tq, m_blocks * block_size]

        # Gather true global indices from candidate map
        topk_idx = _tf_gather_nd(global_candidate_map, local_idx,
                                 batch_dims=3, axis=3)
        # [B, H, Tq, K_eff]

        # Gather selected keys from original k
        selected_k = _tf_gather_nd(k, topk_idx, batch_dims=2, axis=2)
        # [B, H, Tq, K_eff, D]

        # Concatenate q (broadcast over K) with selected keys
        q_for_concat = q.unsqueeze(3)  # [B, H, Tq, 1, D]
        q_bc = q_for_concat.expand(B, H, Tq, K_eff, D)
        topk_pairs = torch.cat([q_bc, selected_k], dim=-1)
        # [B, H, Tq, K_eff, 2D]

        return topk_pairs, topk_idx

    # Phi, tau, and time interpolation
    def compute_phi_tau(self, q: torch.Tensor, k: torch.Tensor, t=None):
        """
        Compute phi (gating), tau (time constant), and time interpolation.

        Args:
            q: [B, H, Tq, D]
            k: [B, H, Tk, D]
            t: time scalar or tensor (supports scalar, [Tq], [B, Tq])

        Returns:
            phi:      [B, H, Tq, K_eff]
            tau:      [B, H, Tq, K_eff]
            t_interp: [B, H, Tq, K_eff]
            topk_idx: [B, H, Tq, K_eff]
        """
        B, H, Tq, _ = q.shape

        pair, topk_idx = self.sparse_topk_pairwise(q, k, K=self.topk)
        K_eff = pair.shape[3]

        # Flatten for out_ncp RNN: [N, 1, 2D] where N = B*H*Tq*K_eff
        flat = pair.reshape(-1, pair.shape[-1])   # [N, 2D]
        flat_3d = flat.unsqueeze(1)                # [N, 1, 2D]

        # Run out_ncp RNN on synthetic length-1 sequence.
        out_raw = self.out_ncp(flat_3d)            # [N, 1, output_size]
        out_raw = out_raw.squeeze(1)                # [N, output_size]

        # Both reshaped to [B, H, Tq, K_eff]
        phi = torch.sigmoid(out_raw).reshape(B, H, Tq, K_eff)
        tau = (F.softplus(out_raw) + self.tau_epsilon).reshape(B, H, Tq, K_eff)

        # Fused Dense(2) reading of the pair tensor
        tab = self.time_ab(pair)          # [B, H, Tq, K_eff, 2]
        t_a = tab[..., :1]                # [B, H, Tq, K_eff, 1]
        t_b = tab[..., 1:]                # [B, H, Tq, K_eff, 1]

        # Broadcast t to [B, 1, Tq, 1, 1]
        if t is None:
            t_expanded = torch.ones(B, 1, Tq, 1, 1, dtype=pair.dtype,
                                    device=pair.device)
        else:
            t_val = t.to(dtype=pair.dtype)
            t_rank = t_val.ndim
            if t_rank == 0:                              # scalar
                t_val = t_val.reshape(1, 1, 1, 1, 1)
            elif t_rank == 1:                            # [Tq]
                t_val = t_val.reshape(1, 1, -1, 1, 1)
            else:                                         # [B, Tq] or broadcastable
                t_val = t_val.reshape(B, 1, Tq, 1, 1)
            t_expanded = t_val.expand(B, 1, Tq, 1, 1)

        # t_interp = sigmoid(-t_a * t + t_b) → squeeze last dim
        t_interp = torch.sigmoid(-t_a * t_expanded + t_b)[..., 0]
        # [B, H, Tq, K_eff]

        return phi, tau, t_interp, topk_idx

    # Extract membrane voltages
    def extract_membrane_voltages(self, gate: _NCPWrapper,
                                  x: torch.Tensor,
                                  state: torch.Tensor = None):
        """
        Measure individual membrane potentials over time.

        Args:
            gate:  _NCPWrapper (has .cell: NCPCell)
            x:     [B, T, D] for sequences or [B, D] for single-step
            state: Optional initial state [B, state_size]

        Returns:
            mmvs:  [B, T, state_size] for sequences, [B, state_size] for single-step
            state: Final state
        """
        B = x.shape[0]

        if state is None:
            state = gate.cell.init_state(B, dtype=x.dtype, device=x.device)

        if x.ndim == 3:  # Sequence input [B, T, D]
            T = x.shape[1]
            mmvs_list = []
            for t in range(T):
                x_t = x[:, t, :]
                _, state = gate.cell(x_t, state)
                mmvs_list.append(state)  # [B, state_size]
            mmvs = torch.stack(mmvs_list, dim=1)  # [B, T, state_size]
        else:  # Single-step [B, D]
            _, state = gate.cell(x, state)
            mmvs = state  # [B, state_size]

        return mmvs, state

    # Cell state computation
    def cell_state(self, q_in: torch.Tensor, k_in: torch.Tensor,
                   v_in: torch.Tensor,
                   qh: torch.Tensor = None, kh: torch.Tensor = None):
        """
        Compute membrane potentials for:
          - sensory projections: q_proj, k_proj, v_proj
          - out_ncp (backbone)

        Reuses pre-computed qh/kh from call() when available to avoid
        redundant RNN passes.

        Args:
            q_in, k_in, v_in: Raw inputs [B, T, d_model]
            qh: Pre-computed multi-head query [B, H, Tq, depth] (optional)
            kh: Pre-computed multi-head key [B, H, Tq, depth] (optional)

        Returns:
            dict: mmvs for 'out_ncp', 'q_proj', 'k_proj', 'v_proj'
        """
        mmvs = {}

        # Compute head-split tensors if not provided
        if qh is None:
            qh = self.split_heads(self.q_proj(q_in))
        if kh is None:
            kh = self.split_heads(self.k_proj(k_in))

        # ---- out_ncp membrane potentials (per head) ----
        pair_all, _ = self.sparse_topk_pairwise(qh, kh, K=self.topk)
        # [B, H, Tq, K_eff, 2D]
        B, H, Tq, K_eff, D2 = pair_all.shape
        sensory_size = self.out_ncp.cell.sensory_size  # = 2*depth

        # Conditionally truncate or pad to match sensory_size
        if D2 > sensory_size:
            pair_all_trunc = pair_all[..., :sensory_size]
        else:
            pad_size = sensory_size - D2
            # PyTorch: pad last dim on the right
            pair_all_trunc = F.pad(pair_all, (0, pad_size), mode='constant', value=0.0)

        # Aggregate top-k: mean over K_eff → [B, H, Tq, sensory_size]
        pair_combined_all = torch.mean(
            pair_all_trunc.reshape(B, H, Tq, K_eff, sensory_size), dim=3
        )

        # Per-head membrane potential extraction
        head_mmvs = []
        for h in range(H):
            mmv, _ = self.extract_membrane_voltages(
                self.out_ncp, pair_combined_all[:, h]
            )
            head_mmvs.append(mmv)

        # Concatenate across heads: [B, Tq, H * state_size]
        mmvs["out_ncp"] = torch.cat(head_mmvs, dim=-1)

        # ---- Sensory projection membrane potentials ----
        mmvs["q_proj"], _ = self.extract_membrane_voltages(self.q_proj, q_in)
        mmvs["k_proj"], _ = self.extract_membrane_voltages(self.k_proj, k_in)
        mmvs["v_proj"], _ = self.extract_membrane_voltages(self.v_proj, v_in)

        return mmvs

    # Forward pass
    def forward(self, inputs, mask=None):
        """
        Forward computation of the NAC layer.

        Input formats:
          - x                    → q=k=v=x,      t=None
          - (x, t)               → q=k=v=x,      t=t
          - (q, k, v)            → q,k,v separate, t=None
          - (q, k, v, t)         → q,k,v separate, t=t

        Args:
            inputs: tensor or tuple as described above
            mask:   optional attention mask [B, Tk] (1=valid, 0=masked)

        Returns:
            Output tensor, or tuple (output, attn_weights, cell_voltages)
            depending on flags.
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
                raise ValueError(f"Unsupported input tuple length: {len(inputs)}")
        else:
            q_in = k_in = v_in = inputs
            t = None

        # ---- Project q, k, v through sensory neurons ----
        q = self.q_proj(q_in)  # [B, T, d_model]
        k = self.k_proj(k_in)
        v = self.v_proj(v_in)

        # ---- Split into multiple heads ----
        qh = self.split_heads(q)   # [B, H, T, depth]
        kh = self.split_heads(k)
        vh = self.split_heads(v)

        B, H, Tq, _ = qh.shape

        # ---- Compute phi, tau, and time interpolation ----
        phi, tau, t_interp, topk_idx = self.compute_phi_tau(qh, kh, t)

        # ---- Solve dynamics based on chosen mode ----
        if self.mode == 'steady':
            attn_logits = phi / tau
        elif self.mode == 'exact':
            attn_logits = (phi / tau) * (1 - torch.exp(-tau * t_interp))
        elif self.mode == 'euler':
            a = torch.zeros_like(phi)
            for _ in range(self.euler_steps):
                increment = self.delta_t * (-tau * a + phi)
                a = a + increment
            attn_logits = a
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # ---- Apply mask ----
        # Broadcast [B, Tk] mask to [B, H, Tq, Tk], then gather per topk_idx.
        if mask is not None:
            mask_f = mask.to(attn_logits.dtype)              # [B, Tk]
            mask_b = mask_f[:, None, None, :]                 # [B, 1, 1, Tk]
            mask_b = mask_b.expand(B, H, Tq, mask.shape[-1]) # [B, H, Tq, Tk]
            mask_gathered = _tf_gather_nd(mask_b, topk_idx,
                                          batch_dims=3, axis=3)
            # [B, H, Tq, K_eff]
            attn_logits = attn_logits * mask_gathered

        # ---- Normalize and apply dropout ----
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # ---- Gather values corresponding to selected keys ----
        vh_topk = _tf_gather_nd(vh, topk_idx, batch_dims=2, axis=2)
        # [B, H, Tq, K_eff, depth]

        # ---- Weighted sum (with optional Riemann sum) ----
        w = attn_weights * t_interp if self.use_riemann_sum else attn_weights
        out_per_head = torch.einsum("bhqk,bhqkd->bhqd", w, vh_topk)
        # [B, H, Tq, depth]

        # ---- Combine heads and project output ----
        combined = self.combine_heads(out_per_head)  # [B, Tq, d_model]
        out = self.out_dense(combined)
        if self.activation is not None:
            out = self.activation(out)

        # ---- Optionally return only the last time step ----
        if not self.return_sequences:
            out = out[:, -1, :]

        # ---- Optionally extract cell voltages ----
        cell_voltages = None
        if self.return_cell_state:
            cell_voltages = self.cell_state(q_in, k_in, v_in, qh=qh, kh=kh)

        # ---- Return based on flags ----
        if self.return_attention and cell_voltages is not None:
            return out, attn_weights, cell_voltages
        elif self.return_attention:
            return out, attn_weights
        elif cell_voltages is not None:
            return out, cell_voltages

        return out

    # Extra representation
    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"topk={self.topk}, mode='{self.mode}', "
            f"euler_steps={self.euler_steps}, sparsity={self.sparsity}, "
            f"delta_t={self.delta_t}, tau_epsilon={self.tau_epsilon}, "
            f"dropout={self.dropout_rate}, use_bias={self.use_bias}, "
            f"use_riemann_sum={self.use_riemann_sum}, "
            f"return_attention={self.return_attention}, "
            f"return_sequences={self.return_sequences}, "
            f"return_cell_state={self.return_cell_state}"

        )