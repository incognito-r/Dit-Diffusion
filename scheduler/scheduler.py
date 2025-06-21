import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):
    """
    Linear (or cosine) noise scheduler for DDPM.

    Forward process:
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

    Reverse process (approximate using noise prediction):
        p(x_{t-1} | x_t) ~ N(mu_t, sigma_t^2 * I)
        where:
            mu_t = (1 / sqrt(alpha_t)) * (x_t - beta_t * noise_pred / sqrt(1 - alpha_bar_t))
            x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
    """

    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 schedule_type='linear', device='cpu'):
        super().__init__()
        self.timesteps = timesteps
        self.device = device

        # Generate betas
        betas = self.get_betas(beta_start, beta_end, schedule_type).to(device)
        alphas = 1. - betas
        alpha_cum_prod = torch.cumprod(alphas, dim=0)
        one_minus_alpha_cum_prod = 1. - alpha_cum_prod

        # Cache all values for sampling
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cum_prod", alpha_cum_prod)
        self.register_buffer("sqrt_alpha_cum_prod", torch.sqrt(alpha_cum_prod))
        self.register_buffer("sqrt_one_minus_alpha_cum_prod", torch.sqrt(one_minus_alpha_cum_prod))
        self.register_buffer("one_minus_alpha_cum_prod", one_minus_alpha_cum_prod)
        self.register_buffer("sqrt_recip_alpha_cum_prod", torch.sqrt(1. / alpha_cum_prod))
        self.register_buffer("sqrt_recipm1_alpha_cum_prod", torch.sqrt(1. / alpha_cum_prod - 1))

    def get_betas(self, beta_start, beta_end, schedule_type):
        if schedule_type == 'linear':
            return torch.linspace(beta_start, beta_end, self.timesteps)
        elif schedule_type == 'cosine':
            s = 0.008
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.timesteps + s) / (1 + s)) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, min=1e-4, max=0.999)

    def add_noise(self, original, noise, t):
        """
        Forward diffusion: q(x_t | x_0)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """

        batch_size = original.size(0)
        shape = [batch_size] + [1] * (original.dim() - 1)

        sqrt_alpha_cum_prod_t = self.sqrt_alpha_cum_prod[t].reshape(shape)
        sqrt_one_minus_alpha_cum_prod_t = self.sqrt_one_minus_alpha_cum_prod[t].reshape(shape)

        return sqrt_alpha_cum_prod_t * original + sqrt_one_minus_alpha_cum_prod_t * noise

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Reverse diffusion: Estimate x0 and sample x_{t-1} ~ p(x_{t-1} | x_t)
        """
        shape = [xt.size(0)] + [1] * (xt.dim() - 1)

        beta_t = self.betas[t].reshape(shape)
        alpha_t = self.alphas[t].reshape(shape)
        alpha_bar_t = self.alpha_cum_prod[t].reshape(shape)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_cum_prod[t].reshape(shape)
        sqrt_recip_alpha_bar_t = self.sqrt_recip_alpha_cum_prod[t].reshape(shape)

        # Estimate x0
        x0 = (xt - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_recip_alpha_bar_t
        x0 = torch.clamp(x0, -1., 1.)

        # Compute mean (mu) of posterior
        mean = (xt - beta_t * noise_pred / sqrt_one_minus_alpha_bar_t) / torch.sqrt(alpha_t)

        # Sample x_{t-1}
        if t == 0:
            return mean, x0
        else:
            alpha_bar_prev = self.alpha_cum_prod[t - 1].reshape(shape)
            variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            sigma = torch.sqrt(variance)
            noise = torch.randn_like(xt)
            return mean + sigma * noise, x0
