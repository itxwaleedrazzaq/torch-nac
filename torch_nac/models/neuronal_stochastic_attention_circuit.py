import torch

class NSAC(torch.nn.Module):
    """
    Neuronal Stochastic Attention Circuit (NSAC) — Pytorch implementation.

    Uses internal Brownian noise realizations to quantify uncertainty.
    - Epistemic: Variance across different stochastic forward passes.
    - Aleatoric: Mean of the predicted internal circuit noise (std^2).
    """

    def __init__(
        self,
        stochastic_model,
        mc_samples: int = 20,
        ood_mean: float = 0.0,
        ood_std: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            stochastic_model: OUWrap-wrapped NAC model that implements the core
                stochastic circuit with Brownian noise.
            mc_samples: No. of Monte Carlo samples during training and inference.
            ood_mean: Mean of the Gaussian perturbation used to generate OOD
                samples during training.
            ood_std: Standard deviation for generating OOD samples via Gaussian
                perturbation during training.
        """
        super().__init__(**kwargs)

        self.stochastic_circuit = stochastic_model
        self.mc_samples = int(mc_samples)
        self.ood_mean = float(ood_mean)
        self.ood_std = float(ood_std)

    # Custom training step
    def compute_loss(self, x, y, loss_fn, training=True):
        """
        
        Args:
            x:        input tensor [B, T, 1] (float32; cast if needed).
            y:        target tensor  [B, 1]  (float32; cast if needed).
            loss_fn:  callable(y_true, [id_samples, ood_samples]) ->
                      (total_loss, nll_loss, reg_loss). Typically an NSACLoss.
            training: bool. TF passes training=True in BOTH train_step and
                      test_step (MC dropout + Brownian noise stay active during
                      evaluation). Pass True to match TF; pass False for a
                      deterministic eval pass.

        Returns:
            (total_loss, nll_loss, reg_loss) — 0-d torch tensors. total_loss
            carries gradients to model parameters; call total_loss.backward().
        """
        x_id = x.to(torch.float32)
        y_id = y.to(torch.float32)

        # Generate OOD samples via Gaussian perturbation for regularization.
        x_ood = x_id + (
            torch.randn(x_id.shape, dtype=x_id.dtype, device=x_id.device) * self.ood_std
            + self.ood_mean
        )

        id_samples = []
        ood_samples = []
        # MC sampling. Both id and ood run with the requested training flag
        for _ in range(self.mc_samples):
            mu_id, std_id = self.stochastic_circuit(x_id, training=training)
            mu_ood, std_ood = self.stochastic_circuit(x_ood, training=training)
            id_samples.append((mu_id, std_id))
            ood_samples.append((mu_ood, std_ood))

        total_loss, nll_loss, reg_loss = loss_fn(y_id, [id_samples, ood_samples])
        return total_loss, nll_loss, reg_loss

    # Forward (NSAC.call)
    def forward(self, inputs, training=None):
        return self.stochastic_circuit(inputs, training=training)

    # Uncertainty decomposition
    def _get_stochastic_components(self, x):
        """
        Internal helper to run multiple Brownian noise realizations and
        calculate the variance decomposition.
        """
        means = []
        variances = []

        for _ in range(self.mc_samples):
            m, s = self.stochastic_circuit(x, training=False)
            means.append(m)
            variances.append((torch.exp(s)) ** 2)   # std^2 = aleatoric variance

        means = torch.stack(means)          # [mc, ...]
        variances = torch.stack(variances)  # [mc, ...]

        # Decomposition: Total = Var(Means) + Mean(Vars)
        mean_prediction = torch.mean(means, dim=0)
        epistemic = torch.var(means, dim=0, unbiased=False)
        aleatoric = torch.mean(variances, dim=0)

        return mean_prediction, aleatoric, epistemic

    def predict(self, x, **kwargs):
        """Return (mean, total_uncertainty) as numpy arrays."""
        mean, aleatoric, epistemic = self._get_stochastic_components(x)
        total_uncertainty = epistemic + aleatoric
        return mean.detach().cpu().numpy(), total_uncertainty.detach().cpu().numpy()

    def predict_with_uncertainty(self, x, **kwargs):
        """Return (mean, aleatoric, epistemic) as numpy arrays."""
        mean, aleatoric, epistemic = self._get_stochastic_components(x)
        return (
            mean.detach().cpu().numpy(),
            aleatoric.detach().cpu().numpy(),
            epistemic.detach().cpu().numpy(),
        )

    # Convenience: expose the stochastic circuit's trainable parameters
    def stochastic_trainable_parameters(self):
        return [p for p in self.stochastic_circuit.parameters() if p.requires_grad]

