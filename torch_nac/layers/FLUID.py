import numpy as np
import torch

from .liquid_attention import LAN
from .hyperconnections import HyperConnection


# PositionalEncoding
class PositionalEncoding(torch.nn.Module):
    """
    Sinusoidal positional encoding (fixed, non-trainable).
    """

    def __init__(self, d_model, max_len: int = 5000, **kwargs):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        # Precompute positional encodings.
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model)
        )
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, ...]  # (1, max_len, d_model) — add batch dim

        self.register_buffer(
            "pe", torch.from_numpy(pe).float()
        )  # (1, max_len, d_model)

    def forward(self, x):
        L = x.shape[1]
        return x + self.pe[:, :L, :]


# Encoder
class Encoder(torch.nn.Module):
    """
    FLUID encoder block: self-attention (LAN) + feed-forward, each wrapped with
    a HyperConnection residual and a post-LayerNorm.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        topk: int = 8,
        delta_t: float = 0.01,
        euler_steps: int = 5,
        enable_hc: bool = True,
        dynamic_hc: bool = True,
        use_sink_gate: bool = True,
        use_pairwise: bool = False,
        expansion_rate: int = 4,
        dropout: float = 0.1,
        return_attention: bool = False,
    ):
        super().__init__()

        self.hc = enable_hc
        self.return_attention = return_attention

        self.attn = LAN(
            d_model=d_model,
            num_heads=num_heads,
            topk=topk,
            delta_t=delta_t,
            euler_steps=euler_steps,
            use_sink_gate=use_sink_gate,
            use_pairwise=use_pairwise,
            return_sequences=True,
            return_attention=return_attention,
        )
        self.drop1 = torch.nn.Dropout(dropout)
        self.enc_norm1 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.hyper_residual1 = HyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            dynamic_hc=dynamic_hc,
            layer_id=1,
        )

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ff_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_dim, d_model),
        )
        self.drop2 = torch.nn.Dropout(dropout)
        self.enc_norm2 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.hyper_residual2 = HyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            dynamic_hc=dynamic_hc,
            layer_id=2,
        )

    def forward(self, x):
        # ---- Self-attention ----
        if self.return_attention:
            attn_out, attn_weights = self.attn(x)
        else:
            attn_out = self.attn(x)
            attn_weights = None
        attn_out = self.drop1(attn_out)
        if self.hc:
            x = self.hyper_residual1([x, attn_out])
        else:
            x = x + attn_out
        x = self.enc_norm1(x)

        # ---- Feed-forward network ----
        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out)
        if self.hc:
            x = self.hyper_residual2([x, ffn_out])
        else:
            x = x + ffn_out
        x = self.enc_norm2(x)

        return x, attn_weights


# Decoder
class Decoder(torch.nn.Module):
    """
    FLUID decoder block: self-attention (LAN) -> cross-attention (LAN over
    encoder output) -> feed-forward, each with a HyperConnection residual and a
    post-LayerNorm.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        topk: int = 8,
        delta_t: float = 0.01,
        euler_steps: int = 5,
        enable_hc: bool = True,
        dynamic_hc: bool = True,
        use_sink_gate: bool = True,
        use_pairwise: bool = False,
        expansion_rate: int = 4,
        dropout: float = 0.1,
        return_attention: bool = False,
    ):
        super().__init__()

        self.hc = enable_hc
        self.return_attention = return_attention

        # Self-attention
        self.self_attn = LAN(
            d_model=d_model,
            num_heads=num_heads,
            topk=topk,
            delta_t=delta_t,
            euler_steps=euler_steps,
            use_sink_gate=use_sink_gate,
            use_pairwise=use_pairwise,
            return_sequences=True,
            return_attention=return_attention,
        )
        self.dec_norm1 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.drop1 = torch.nn.Dropout(dropout)
        self.hyper_residual1 = HyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            dynamic_hc=dynamic_hc,
            layer_id=1,
        )

        self.cross_attn = LAN(
            d_model=d_model,
            num_heads=num_heads,
            delta_t=delta_t,
            topk=topk,
            use_sink_gate=use_sink_gate,
            use_pairwise=use_pairwise,
            return_sequences=True,
            return_attention=return_attention,
        )
        self.dec_norm2 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.drop2 = torch.nn.Dropout(dropout)
        self.hyper_residual2 = HyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            dynamic_hc=dynamic_hc,
            layer_id=2,
        )

        # Feed-forward network
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ff_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_dim, d_model),
        )
        self.drop3 = torch.nn.Dropout(dropout)
        self.dec_norm3 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.hyper_residual3 = HyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            dynamic_hc=dynamic_hc,
            layer_id=3,
        )

    def forward(self, x, enc_out):
        # ---- Self-attention ----
        if self.return_attention:
            sa, sa_weights = self.self_attn(x)
        else:
            sa = self.self_attn(x)
            sa_weights = None

        sa = self.drop1(sa)
        if self.hc:
            x = self.hyper_residual1([x, sa])
        else:
            x = x + sa
        x = self.dec_norm1(x)

        # ---- Cross-attention: decoder attending over encoder output ----
        if self.return_attention:
            ca, ca_weights = self.cross_attn([x, enc_out, enc_out])
        else:
            ca = self.cross_attn([x, enc_out, enc_out])
            ca_weights = None
        ca = self.drop2(ca)
        if self.hc:
            x = self.hyper_residual2([x, ca])
        else:
            x = x + ca
        x = self.dec_norm2(x)

        # ---- Feed-forward network ----
        ffn_out = self.ffn(x)
        ffn_out = self.drop3(ffn_out)
        if self.hc:
            x = self.hyper_residual3([x, ffn_out])
        else:
            x = x + ffn_out
        x = self.dec_norm3(x)

        dec_weights = {
            "self_attention": sa_weights,
            "cross_attention": ca_weights,
        }

        return x, dec_weights


# FLUID
class FLUID(torch.nn.Module):
    """
    FLUID: Flexible Unified Information Dynamics.

    Input projection (Dense) → positional encoding → stacked Encoders → stacked
    Decoders (which cross-attend to the final encoder output). Optionally
    returns per-layer attention weights for analysis.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        topk: int = 8,
        euler_steps: int = 5,
        delta_t: float = 0.01,
        enable_hc: bool = True,
        dynamic_hc: bool = True,
        expansion_rate: int = 4,
        use_sink_gate: bool = True,
        num_layers: int = 1,
        dropout: float = 0.0,
        max_len: int = 1000,
        use_pairwise: bool = False,
        return_attention: bool = False,
        input_dim: int = None,
    ):
        """
        Args:
        - d_model: Dimension of the model (embedding size)
        - num_heads: Number of attention heads
        - ff_dim: Dimension of the feed-forward network
        - topk: Number of top connections to keep in LAN
        - delta_t: fixed time-step for LAN
        - euler_steps: Number of Euler steps for LAN
        - enable_hc: Whether to use hyper-connections
        - dynamic_hc: Whether hyper-connections are dynamic (Liquid)
        - expansion_rate: How many past layers to connect to in hyper-connections
        - use_sink_gate: Whether to use sink gate in LAN
        - num_layers: Number of encoder and decoder layers
        - dropout: Dropout rate
        - max_len: Maximum sequence length for positional encoding
        - use_pairwise: Whether to use pairwise attention in LAN
        - return_attention: Whether to return attention weights for analysis
        - input_dim: (PyTorch only) feature dim of the raw input feeding the
              input-projection Dense. Defaults to d_model.
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.hc = enable_hc
        self.return_attention = return_attention

        # Input projection.
        self.input_dim = int(input_dim) if input_dim is not None else int(d_model)
        self.embedding = torch.nn.Linear(self.input_dim, d_model)

        # Positional encoding (fixed for time-series order awareness)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Multi-layer encoder and decoder.
        self.encoders = torch.nn.ModuleList([
            Encoder(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                topk=topk,
                delta_t=delta_t,
                euler_steps=euler_steps,
                use_sink_gate=use_sink_gate,
                use_pairwise=use_pairwise,
                expansion_rate=expansion_rate,
                enable_hc=enable_hc,
                dynamic_hc=dynamic_hc,
                dropout=dropout,
                return_attention=return_attention,
            )
            for _ in range(num_layers)
        ])

        self.decoders = torch.nn.ModuleList([
            Decoder(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                topk=topk,
                delta_t=delta_t,
                euler_steps=euler_steps,
                use_sink_gate=use_sink_gate,
                use_pairwise=use_pairwise,
                expansion_rate=expansion_rate,
                enable_hc=enable_hc,
                dynamic_hc=dynamic_hc,
                dropout=dropout,
                return_attention=return_attention,
            )
            for _ in range(num_layers)
        ])

        # (GlorotUniform kernel, zeros bias)
        self._init_linear_parameters()

    def _init_linear_parameters(self):
        # Input projection
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.zeros_(self.embedding.bias)

        # FFN Dense layers in each encoder / decoder
        for module in self.encoders:
            self._init_ffn(module.ffn)
        for module in self.decoders:
            self._init_ffn(module.ffn)

    @staticmethod
    def _init_ffn(ffn):
        # ffn = Sequential(Linear(d_model, ff_dim), ReLU, Linear(ff_dim, d_model))
        torch.nn.init.xavier_uniform_(ffn[0].weight)
        torch.nn.init.zeros_(ffn[0].bias)
        torch.nn.init.xavier_uniform_(ffn[2].weight)
        torch.nn.init.zeros_(ffn[2].bias)

    def forward(self, x):
        """
        Args:
          x : (B, L, input_dim) raw input.
        Returns:
          - dec_out : (B, L, d_model) if return_attention is False
          - (dec_out, {'encoder_attention': [...], 'decoder_attention': [...]})
            if return_attention is True. Each entry is the per-layer attention
            structure returned by Encoder/Decoder (None at non-attention layers).
        """
        x = self.embedding(x)          # Project input to d_model
        x = self.pos_encoder(x)        # Positional encoding for time-series generalization
        enc_out = x                    # Multi-layer encoder

        # Attention weights
        enc_weights = []
        dec_weights = []
        for encoder in self.encoders:
            enc_out, enc_weight = encoder(enc_out)
            enc_weights.append(enc_weight)

        # Multi-layer decoder (using input sequence for both; adjust if needed
        # for autoregressive). dec_out starts from the post-PE input x.
        dec_out = x
        for decoder in self.decoders:
            dec_out, dec_weight = decoder(dec_out, enc_out)
            dec_weights.append(dec_weight)

        if self.return_attention:
            return dec_out, {
                "encoder_attention": enc_weights,
                "decoder_attention": dec_weights,
            }
        else:
            return dec_out

