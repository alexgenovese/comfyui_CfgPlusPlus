# Credits:
# CFG++: https://cfgpp-diffusion.github.io/
# DDIM implementation: https://github.com/JettHu/ComfyUI-TCD
# License: GPLv3, see LICENSE
import torch
import logging
from comfy.k_diffusion.sampling import default_noise_sampler
from comfy.samplers import KSAMPLER
from tqdm.auto import trange

@torch.no_grad()
def sample_ddim(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=0.0, cfgpp=False, ud=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    # From comfy/model_sampling.py: sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    alpha_prod_s = 1 / (1 + sigmas ** 2)
    beta_prod_s = 1 - alpha_prod_s
    for i in trange(len(sigmas) - 1, disable=disable):
        # Calling model(...) will update ud.u_denoised with uncond_denoised through post_cfg_function
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        eps_d = ud.u_denoised if cfgpp else denoised
        eps = (x - eps_d) / sigmas[i]
        if eta > 0 and sigmas[i + 1] > 0:
            s_up = eta * ((beta_prod_s[i + 1]) / (beta_prod_s[i])).sqrt() * (1 - (alpha_prod_s[i] / alpha_prod_s[i + 1])).sqrt()
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            denoised =  alpha_prod_s[i + 1].sqrt() * denoised + (beta_prod_s[i + 1] - s_up ** 2).sqrt() * eps + noise * s_up
        else:
            denoised = alpha_prod_s[i + 1].sqrt() * denoised + beta_prod_s[i + 1].sqrt() * eps

        x = denoised * (sigmas[i + 1] ** 2 + 1).sqrt()

    return x

# Uses same denoised calculation as DPM++ 2M but performs DDIM at the end
@torch.no_grad()
def sample_ddim_2m(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, cfgpp=False, ud=None, eta=0):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised, old_u_denoised = None, None
    # From comfy/model_sampling.py: sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    alpha_prod_s = 1 / (1 + sigmas ** 2)
    beta_prod_s = 1 - alpha_prod_s
    for i in trange(len(sigmas) - 1, disable=disable):
        # Calling model(...) will update ud.u_denoised with uncond_denoised through post_cfg_function
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        eps_d = ud.u_denoised if cfgpp else denoised
        if old_denoised is not None and sigmas[i + 1] > 0:
            _d = denoised
            _ud = eps_d
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            r = (t - t_fn(sigmas[i - 1])) / (t_next - t)
            rf = lambda d, pd: (1 + 1 / (2 * r)) * d - (1 / (2 * r)) * pd
            denoised = rf(denoised, old_denoised)
            eps_d = rf(eps_d, old_u_denoised)
            old_denoised = _d
            old_u_denoised = _ud
        else:
            # Replace line below with commented out and turn cfg++ off for funny results
            #old_u_denoised = ud.u_denoised
            old_u_denoised = eps_d
            old_denoised = denoised

        eps = (x - eps_d) / sigmas[i]
        if eta > 0 and sigmas[i + 1] > 0:
            s_up = eta * ((beta_prod_s[i + 1]) / (beta_prod_s[i])).sqrt() * (1 - (alpha_prod_s[i] / alpha_prod_s[i + 1])).sqrt()
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            denoised =  alpha_prod_s[i + 1].sqrt() * denoised + (beta_prod_s[i + 1] - s_up ** 2).sqrt() * eps + noise * s_up
        else:
            denoised = alpha_prod_s[i + 1].sqrt() * denoised + beta_prod_s[i + 1].sqrt() * eps

        x = denoised * (sigmas[i + 1] ** 2 + 1).sqrt()

    return x

class CFGPP:
    def __init__(self, *args, **kwargs):
        # I am not sure if that is really needed
        super().__init__(*args, **kwargs)
        # Will be assigned to uncond_denoised later
        self.u_denoised = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                # Becomes just DDIM scheduler when False
                "cfgpp_enabled": ("BOOLEAN", {"default": True}),
                # Setting it over 0 makes it behave like TCD sampler (does not work with CFG++ properly)
                "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # ddim - from the CFG++ paper
                # ddim_2m - ddim with DPM++ 2M denoise step
                "sampler": (["ddim", "ddim_2m"],)
            }
        }

    RETURN_TYPES = ("MODEL", "SAMPLER")
    FUNCTION = "patch"
    # It requires "model" category so "model" input becomes available
    CATEGORY = "advanced/model"

    def patch(self, model, cfgpp_enabled, eta, sampler):
        m = model.clone()
        sample = sample_ddim if sampler == "ddim" else sample_ddim_2m
        def post_cfg_function(args):
            self.u_denoised = args['uncond_denoised']
            return args['denoised']
        # uncond_denoised is not exposed to sampler function directly
        # Get it by hooking post_cfg_function
        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return (m, KSAMPLER(sample, extra_options={'eta': eta, 'cfgpp': cfgpp_enabled, 'ud': self}, inpaint_options={}))

NODE_CLASS_MAPPINGS = {
    "CFG++": CFGPP
}
