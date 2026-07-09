import torch

class HyperConnection(torch.nn.Module):
    """
    HyperConnection (Liquid / static) — PyTorch port.

    A residual mixing block from https://arxiv.org/abs/2409.19606. It mixes a
    "layer input" x with a "layer output" x_o using learned alpha (mixing
    coefficients over past-layer copies) and beta (scaling of x_o) terms, which
    can be either static (learned scalars) or dynamic (input-conditioned via a
    LayerNorm + tanh gated linear map).
    """

    def __init__(self, d_model, expansion_rate, layer_id, dynamic_hc, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.expansion_rate = expansion_rate
        self.layer_id = layer_id
        self.dynamic_hc = dynamic_hc

        # ----- static beta -----
        self.static_beta = torch.nn.Parameter(
            torch.ones(self.expansion_rate), requires_grad=True
        )

        # ----- static alpha -----
        init_alpha0 = torch.zeros((self.expansion_rate, 1), dtype=torch.float32)
        init_alpha0[self.layer_id % self.expansion_rate, 0] = 1.0
        eye = torch.eye(self.expansion_rate, dtype=torch.float32)
        init_alpha = torch.cat([init_alpha0, eye], dim=1)  # (E, E+1)

        self.static_alpha = torch.nn.Parameter(init_alpha.clone(), requires_grad=True)

        if self.dynamic_hc:
            self.dynamic_hc_alpha_fn = torch.nn.Parameter(
                torch.zeros(self.d_model, self.expansion_rate + 1),
                requires_grad=True,
            )
            self.dynamic_hc_alpha_scale = torch.nn.Parameter(
                torch.tensor([0.01], dtype=torch.float32), requires_grad=True
            )
            self.dynamic_hc_beta_fn = torch.nn.Parameter(
                torch.zeros(self.d_model), requires_grad=True
            )
            self.dynamic_hc_beta_scale = torch.nn.Parameter(
                torch.tensor([0.01], dtype=torch.float32), requires_grad=True
            )
            self.layer_norm = torch.nn.LayerNorm(
                normalized_shape=self.d_model, eps=1e-3, elementwise_affine=True
            )

    def forward(self, inputs):
        """
        Args:
          inputs: a 2-element list/tuple [x, x_o]
            x   : (B, L, d_model) — layer input (the residual stream entering
                  this block; tiled across the expansion_rate "past layers").
            x_o : (B, L, d_model) — layer output (e.g. attention/FFN output)
                  to be mixed into the residual stream.
        Returns:
          new_x : (B, L, d_model)
        """
        x, x_o = inputs
        x_exp = x.unsqueeze(2).expand(
            x.shape[0], x.shape[1], self.expansion_rate, x.shape[2]
        )

        if self.dynamic_hc:
            norm_x = self.layer_norm(x_o)
            wc = torch.tanh(torch.matmul(norm_x, self.dynamic_hc_alpha_fn)) \
                * self.dynamic_hc_alpha_scale
            alpha = wc[:, :, None, :] + self.static_alpha[None, None, :, :]
        else:
            alpha = self.static_alpha[None, None, :, :]

        if self.dynamic_hc:
            dc = torch.tanh(
                torch.matmul(
                    norm_x,
                    self.dynamic_hc_beta_fn.reshape(self.d_model, 1),
                )
            ) * self.dynamic_hc_beta_scale
            beta = dc + self.static_beta[None, None, :]
        else:
            # beta: (1, 1, expansion_rate) — broadcasts on use.
            beta = self.static_beta[None, None, :]

        # mix_x: (B, L, expansion_rate, d_model)
        mix_x = torch.matmul(alpha[:, :, :, : self.expansion_rate], x_exp)
        # beta_sum: (B, L, 1)
        beta_sum = torch.sum(beta, dim=-1, keepdim=True)
        # new_x: (B, L, d_model)
        new_x = x_o * beta_sum + torch.sum(mix_x, dim=2)

        return new_x
