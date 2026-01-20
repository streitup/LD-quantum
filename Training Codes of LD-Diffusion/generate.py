# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import torch.nn.functional as F
import dnnlib
from torch_utils import distributed as dist
from training.pos_embedding import Pos_Embedding

# Optional dependency: diffusers for latent VAE decoding.
# Import lazily to avoid hard dependency when not using --on_latents.
try:
    from diffusers import AutoencoderKL  # type: ignore
except Exception:
    AutoencoderKL = None


def random_patch(images, patch_size, resolution):
    device = images.device

    pos_shape = (images.shape[0], 1, patch_size, patch_size)
    x_pos = torch.ones(pos_shape)
    y_pos = torch.ones(pos_shape)
    x_start = np.random.randint(resolution - patch_size)
    y_start = np.random.randint(resolution - patch_size)

    x_pos = x_pos * x_start + torch.arange(patch_size).view(1, -1)
    y_pos = y_pos * y_start + torch.arange(patch_size).view(-1, 1)

    x_pos = (x_pos / resolution - 0.5) * 2.
    y_pos = (y_pos / resolution - 0.5) * 2.

    # Add x and y additional position channels
    images_patch = images[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
    images_pos = torch.cat([x_pos.to(device), y_pos.to(device)], dim=1)

    return images_patch, images_pos

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, latents_pos, mask_pos, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # img_channel = latents.shape[1]
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Detect whether the loaded network expects positional inputs (Patch_EDMPrecond).
    # If so, we will pass x_pos via keyword; otherwise, we NEVER pass latents_pos.
    is_patch_model = ('Patch_EDMPrecond' in net.__class__.__name__)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Prepare per-sample sigma vector (shape [B]) for network forward.
        B = x_hat.shape[0]
        if isinstance(t_hat, torch.Tensor):
            if t_hat.ndim == 0:
                t_hat_b = t_hat.repeat(B)
            elif t_hat.shape[0] == B:
                t_hat_b = t_hat
            else:
                t_hat_b = t_hat.reshape(-1).repeat(B)[:B]
        else:
            t_hat_b = torch.full((B,), float(t_hat), dtype=torch.float64, device=latents.device)

        # Euler step.
        # For Patch_EDMPrecond models, ALWAYS pass x_pos (masked to zeros when mask_pos=True).
        if is_patch_model:
            # Patch_EDMPrecond signature: (x, sigma, x_pos=None, class_labels=None, ...)
            denoised = net(x_hat, t_hat_b, x_pos=latents_pos, class_labels=class_labels, force_fp32=True).to(torch.float64)
        else:
            # Standard EDMPrecond signature: (x, sigma, class_labels=None, ...)
            denoised = net(x_hat, t_hat_b, class_labels=class_labels, force_fp32=True).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            # Prepare per-sample sigma vector for t_next.
            if isinstance(t_next, torch.Tensor):
                if t_next.ndim == 0:
                    t_next_b = t_next.repeat(B)
                elif t_next.shape[0] == B:
                    t_next_b = t_next
                else:
                    t_next_b = t_next.reshape(-1).repeat(B)[:B]
            else:
                t_next_b = torch.full((B,), float(t_next), dtype=torch.float64, device=latents.device)

            if is_patch_model:
                denoised = net(x_next, t_next_b, x_pos=latents_pos, class_labels=class_labels, force_fp32=True).to(torch.float64)
            else:
                denoised = net(x_next, t_next_b, class_labels=class_labels, force_fp32=True).to(torch.float64)

            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels, force_fp32=True).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels, force_fp32=True).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

#----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--resolution',              help='Sample resolution', metavar='INT',                                 type=int, default=64)
@click.option('--embed_fq',                help='Positional embedding frequency', metavar='INT',                    type=int, default=0)
@click.option('--mask_pos',                help='Mask out pos channels', metavar='BOOL',                            type=bool, default=False, show_default=True)
@click.option('--on_latents',              help='Generate with latent vae', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--vae_local_dir',           help='Local directory for AutoencoderKL.from_pretrained when running offline', metavar='DIR', type=str, default=None)
@click.option('--latents_fallback',        help='Fallback when VAE is unavailable: save latents as RGB using first 3 channels', metavar='none|rgb_first3', type=click.Choice(['none','rgb_first3']), default='rgb_first3', show_default=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--device', 'device_str',    help='Choose device for sampling', metavar='cpu|cuda',                   type=click.Choice(['cpu','cuda']), default='cuda', show_default=True)

# patch options
@click.option('--x_start',                 help='Sample resolution', metavar='INT',                                 type=int, default=0)
@click.option('--y_start',                 help='Sample resolution', metavar='INT',                                 type=int, default=0)
@click.option('--image_size',                help='Sample resolution', metavar='INT',                                 type=int, default=None)
@click.option('--output_size',              help='Final output image size (post-decoding resize)', metavar='INT',    type=int, default=None)

@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, resolution, on_latents, embed_fq, mask_pos, vae_local_dir, latents_fallback, x_start, y_start, image_size, output_size, outdir, subdirs, seeds, class_idx, max_batch_size, device_str='cuda', **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    # Initialize distributed and resolve device.
    dist.init()
    device = torch.device('cuda' if (device_str == 'cuda' and torch.cuda.is_available()) else 'cpu')
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    # Infer model input requirements for sampling shape alignment.
    # QuantumTransformerDenoiser expects latent inputs with fixed img_resolution and in_channels as trained.
    net_img_res = getattr(net, 'img_resolution', None)
    net_in_ch   = getattr(net, 'img_channels', None)

    net_out_ch  = getattr(net, 'out_channels', None)
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    decode_latents = False
    latent_scale_factor = 0.18215
    if on_latents:
        # Strict behavior: when sampling on latents, VAE must be available and loadable.
        if AutoencoderKL is None:
            raise RuntimeError("on_latents=1 需要 diffusers.AutoencoderKL 可用；请安装 diffusers/huggingface_hub，或通过 --vae_local_dir 指定本地权重目录。")
        try:
            load_target = vae_local_dir if (vae_local_dir is not None and len(vae_local_dir) > 0) else "stabilityai/sd-vae-ft-ema"
            img_vae = AutoencoderKL.from_pretrained(load_target).to(device)
            img_vae.eval()
            set_requires_grad(img_vae, False)
            decode_latents = True
        except Exception as e:
            raise RuntimeError(f"无法加载 AutoencoderKL：{e}。请确保可联网或将完整权重置于 --vae_local_dir 指向的本地目录（需含 config.json 与 pytorch_model.bin）。")

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)

        image_channel = 3
        if image_size is None:
            image_size = resolution
        # If sampling on latents, force latent shape to match training defaults or net's declared shape.
        if on_latents:
            # Default latent settings: 4 channels at 32x32 with scale factor 0.18215.
            # IMPORTANT: For Patch_EDMPrecond models, the network was trained with latents (C=4) + pos (C=2) concatenated inside forward.
            # Therefore, the input noise 'x' must have 'out_channels' channels (typically 4), NOT the wrapper's 'img_channels' (which may include pos channels, e.g., 6).
            is_patch_model = ('Patch_EDMPrecond' in net.__class__.__name__)
            image_channel = int(net_out_ch) if isinstance(net_out_ch, int) else 4
            image_size    = int(net_img_res) if isinstance(net_img_res, int) else 32
            resolution    = image_size
            x_start       = 0

        x_pos = torch.arange(x_start, x_start+image_size).view(1, -1).repeat(image_size, 1)
        y_pos = torch.arange(y_start, y_start+image_size).view(-1, 1).repeat(1, image_size)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
        latents_pos = torch.stack([x_pos, y_pos], dim=0).to(device)
        latents_pos = latents_pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if mask_pos: latents_pos = torch.zeros_like(latents_pos)
        if embed_fq > 0:
            pos_embed = Pos_Embedding(num_freqs=embed_fq)
            latents_pos = pos_embed(latents_pos)

        latents = rnd.randn([batch_size, image_channel, image_size, image_size], device=device)
        # rnd = StackedRandomGenerator(device, batch_seeds)
        # latents = rnd.randn([batch_size, 3, 64, 64], device=device)
        # latents, latents_pos = random_patch(latents, 16, 64)

        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        images = sampler_fn(net, latents, latents_pos, mask_pos, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        if on_latents:
            images = 1 / latent_scale_factor * images
            # With strict behavior, decode_latents must be True here; otherwise earlier we raised.
            images = img_vae.decode(images.float()).sample

        # Optionally resize final images to requested output_size.
        if output_size is not None:
            h, w = images.shape[-2], images.shape[-1]
            if (h != output_size) or (w != output_size):
                images = F.interpolate(images, size=(output_size, output_size), mode='bilinear', align_corners=False)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
