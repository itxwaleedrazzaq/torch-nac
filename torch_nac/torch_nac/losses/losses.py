import math
import torch


class NSACLoss:
    """Stochastic attention loss: Gaussian NLL on the MC-mean prediction
    plus an epistemic regularizer that penalizes in-distribution variance
    relative to out-of-distribution variance.

    """

    def __init__(self, lambda_reg=0.5, epsilon=1e-3, name="NSAC-Loss"):
        self.lambda_reg = float(lambda_reg)
        self.epsilon = float(epsilon)
        self.name = name

    def __call__(self, y_true, y_pred):
        """
        y_pred = [id_samples, ood_samples]
        Each is a list of tuples (mu, std) — one per MC sample.
        """
        id_samples, ood_samples = y_pred

        # Stack MC samples along a new leading axis.
        mu_id = torch.stack([p[0] for p in id_samples])      # [mc, ...]
        std_id = torch.stack([p[1] for p in id_samples])     # [mc, ...]
        mu_ood = torch.stack([p[0] for p in ood_samples])    # [mc, ...]

        # NLL using mean prediction.
        var_id = (torch.exp(std_id)) ** 2 + self.epsilon     # [mc, ...]
        mu_mean = torch.mean(mu_id, dim=0)                   # [...]
        var_mean = torch.mean(var_id, dim=0)                 # [...]

        # nll = 0.5 * ( log(2*pi) + log(var_mean) + (y - mu)^2 / var_mean )
        nll_loss = 0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(var_mean)
            + (y_true - mu_mean) ** 2 / var_mean
        )
        nll_loss = torch.mean(nll_loss)

        # Epistemic regularizer.
        epi_id = torch.mean(torch.var(mu_id, dim=0, unbiased=False))
        epi_ood = torch.mean(torch.var(mu_ood, dim=0, unbiased=False))
        reg_loss = torch.log(1.0 + (epi_id / (epi_ood + self.epsilon)))

        total_loss = nll_loss + self.lambda_reg * reg_loss

        return total_loss, nll_loss, reg_loss


class GaussianNLL:
    """Gaussian negative log-likelihood for a (mu, log_std) prediction.
    """

    def __init__(self, epsilon=1e-2, name="gaussian_nll"):
        self.epsilon = float(epsilon)
        self.name = name

    def __call__(self, y_true, y_pred):
        mu, log_std = y_pred

        # assume std_id is log(sigma)
        log_std = torch.clamp(log_std, -10.0, 5.0)
        var = torch.exp(2.0 * log_std)

        nll = 0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(var)
            + (y_true - mu) ** 2 / var
        )

        # returns (reduce_mean(nll), reduce_mean(nll), constant(0.0))
        return torch.mean(nll), torch.mean(nll), torch.tensor(0.0, dtype=torch.float32)


