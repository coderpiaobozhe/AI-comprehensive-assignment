import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_named_beta_schedule(schedule_name='linear', num_diffusion_timesteps=1000):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return  betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
        
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianDiffusion(nn.Module):
    def __init__(self, dtype, model, betas, w, v, device):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device)
        self.model.dtype = self.dtype
        self.betas = torch.tensor(betas,dtype=torch.float64)
        self.w = w
        self.v = v
        self.T = len(betas)
        self.device = device
        self.alphas = 1 - self.betas
        self.log_alphas = torch.log(self.alphas)
        
        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim = 0)
        self.alphas_bar = torch.exp(self.log_alphas_bar)
        # self.alphas_bar = torch.cumprod(self.alphas, dim = 0)
        
        self.log_alphas_bar_prev = F.pad(self.log_alphas_bar[:-1],[1,0],'constant',0)
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev)
        self.log_one_minus_alphas_bar_prev = torch.log(1.0 - self.alphas_bar_prev)
        # self.alphas_bar_prev = F.pad(self.alphas_bar[:-1],[1,0],'constant',1)

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas)
        # self.sqrt_alphas = torch.sqrt(self.alphas)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar)
        # self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar)
        
        # calculate parameters for q(x_{t-1}|x_t,x_0)
        # log calculation clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * torch.exp(self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.log_tilde_betas_clipped = torch.log(torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0))
        self.mu_coef_x0 = self.betas * torch.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.mu_coef_xt = torch.exp(0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
    
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar)
        # self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar)
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)
    @staticmethod
    def _extract(coef, t, x_shape):
        """
        input:

        coef : an array
        t : timestep
        x_shape : the shape of tensor x that has K dims(the value of first dim is batch size)

        output:

        a tensor of shape [batchsize,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]

        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        chosen = coef[t]
        chosen = chosen.to(t.device)
        return chosen.reshape(neo_shape)

    def q_mean_variance(self, x_0, t):
        """
        calculate the parameters of q(x_t|x_0)
        """
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var = self._extract(1.0 - self.sqrt_alphas_bar, t, x_0.shape)
        return mean, var
    
    def q_sample(self, x_0, t):
        """
        sample from q(x_t|x_0)
        """
        eps = torch.randn_like(x_0, requires_grad=False)
        return self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 \
            + self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps, eps
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        calculate the parameters of q(x_{t-1}|x_t,x_0)
        """
        posterior_mean = self._extract(self.mu_coef_x0, t, x_0.shape) * x_0 \
            + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        posterior_var_max = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(self.log_tilde_betas_clipped, t, x_t.shape)
        log_posterior_var_max = self._extract(torch.log(self.betas), t, x_t.shape)
        log_posterior_var = self.v * log_posterior_var_max + (1 - self.v) * log_posterior_var_min
        neo_posterior_var = torch.exp(log_posterior_var)
        return posterior_mean, posterior_var_max, neo_posterior_var
    
    def p_mean_variance(self, x_t, t, model_kwargs = None):
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape = model_kwargs['cemb'].shape
        pred_eps_cond = self.model(x_t, t, **model_kwargs)
        model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        logits = pred_eps_cond - pred_eps_uncond
        probs = F.softmax(logits,dim=1).reshape(B,-1)
        log_probs = torch.log(probs)
        entropy_scale = (-log_probs * probs).sum(dim=-1).reshape(B,-1)
        weight = self.w * entropy_scale
        pred_eps = pred_eps_cond + weight * (logits)
        # pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        pred_x_0 = self._predict_x0_from_eps(x_t, t, pred_eps).clamp(-1, 1)

        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"
        assert torch.isnan(pred_x_0).int().sum() == 0, f"nan in tensor pred_x_0 when t = {t[0]}"
        if(t[0] == 0):
            print(x_t)
        p_mean, _, p_var = self.q_posterior_mean_variance(pred_x_0, x_t, t)
        return p_mean, p_var

    def _predict_x0_from_eps(self,x_t, t, eps):
        return self._extract(coef = self.sqrt_recip_alphas_bar, t = t, x_shape = x_t.shape) \
            * (x_t - self._extract(coef = self.sqrt_one_minus_alphas_bar, t = t, x_shape = x_t.shape) * eps)
        # return self._extract(coef = self.sqrt_recip_alphas_bar,t = t, x_shape = x_t.shape) * x_t \
        #    - self._extract(coef = self.sqrt_recipm1_alphas_bar,t = t,x_shape = eps.shape) * eps

    def p_sample(self, x_t, t, model_kwargs = None):
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        mean, var = self.p_mean_variance(x_t , t, model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise
    def sample(self, shape, model_kwargs = None):
        """
        sample images from p_{theta}
        """
        if model_kwargs == None:
            model_kwargs = {}
        x_t = torch.randn(shape, device = self.device)
        for t in reversed(range(self.T)):
            tlist = torch.ones([x_t.shape[0]], device = self.device) * t
            x_t = self.p_sample(x_t, tlist, model_kwargs)
            x_t = torch.clamp(x_t, -1, 1)
            # assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor when t = {t}"
        # return x_t
        return torch.clamp(x_t, -1, 1) 
    def trainloss(self, x_0, model_kwargs=None):
        """
        calculate the loss of denoising diffusion probabilistic model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t = torch.randint(self.T, size = (x_0.shape[0],), device=self.device)
        x_t, eps = self.q_sample(x_0, t)
        loss = F.mse_loss(self.model(x_t, t, **model_kwargs), eps, reduction='sum')
        return loss