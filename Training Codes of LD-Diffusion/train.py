# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

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

@click.command()

#Patch options
@click.option('--real_p',        help='Full size image ratio', metavar='INT',                       type=click.FloatRange(min=0, max=1), default=0.5, show_default=True)
@click.option('--train_on_latents',      help='Training on latent embeddings', metavar='BOOL',      type=bool, default=False, show_default=True)
@click.option('--progressive',      help='Training on latent embeddings', metavar='BOOL',           type=bool, default=False, show_default=True)

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm|quantum_transformer',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm', 'quantum_transformer']), default='ncsnpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm|pedm',  type=click.Choice(['vp', 've', 'edm', 'pedm']), default='edm', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)
@click.option('--implicit_mlp',  help='encoding coordbefore sending to the conv', metavar='BOOL',   type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--device',        help='Torch device to use (e.g., cuda, cuda:2, cpu).', metavar='STR', type=str, default='cuda', show_default=True)
@click.option('--grad-clip-norm', help='Max gradient norm for clipping (None to disable).', metavar='FLOAT', type=float)
# Quantum Transformer model options (active only when --arch=quantum_transformer).
@click.option('--model-dim',      help='Transformer model dimension (DiT tokens_384).', metavar='INT', type=click.IntRange(min=1), default=384, show_default=True)
@click.option('--heads',          help='Transformer heads (for classical backbone; quantum attn is single-head internally).', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--layers',         help='Number of Transformer blocks.', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--patch-size',     help='Patch size for 2D patchify/unpatchify.', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--quantum-n-qubits', help='Number of qubits for QSANN (fixed 6 for 64-d amplitude encoding).', metavar='INT', type=click.IntRange(min=1), default=6, show_default=True)
@click.option('--quantum-depth',  help='Quantum circuit depth (Q_DEPTH).', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--quantum-encoding', help='Quantum encoding scheme.', metavar='STR', type=click.Choice(['amplitude', 'angle']), default='amplitude', show_default=True)
@click.option('--qk-dim',         help='Q/K projection dimension from quantum measurement.', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--quantum-attn-dropout', help='Dropout applied to quantum attention output.', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.1, show_default=True)
@click.option('--quantum-attn-gate-init', help='Initial value for attention residual gate.', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.5, show_default=True)
@click.option('--force-fp32-attn', help='Force quantum attention computation in FP32 under AMP.', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--attn-type', help='Attention type for quantum_transformer backbone', metavar='quantum|classic', type=click.Choice(['quantum','classic']), default='quantum', show_default=True)

# Quantum attention adapter options for classical UNet (SongUNet/DhariwalUNet).
@click.option('--quantum-attn-in-unet', help='Enable QSANN-based attention replacement inside classical UNet blocks (SongUNet/DhariwalUNet).', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--quantum-adapter', help='Adapter path "module:Class" to import (default: QuantumTransformer.QSANNAdapter:QSANNAdapter).', metavar='STR', type=str, default='QuantumTransformer.QSANNAdapter:QSANNAdapter', show_default=True)
@click.option('--quantum-qk-norm', help='Normalization for q/k projection inside adapter.', metavar='STR', type=click.Choice(['none', 'layernorm']), default='layernorm', show_default=True)
@click.option('--quantum-tau', help='RBF attention temperature (if set, overrides adapter default).', metavar='FLOAT', type=float)
@click.option('--quantum-tau-trainable', help='Whether the RBF temperature is trainable via softplus.', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--quantum-attn-chunk', help='Chunk size for QSANN attention batched simulation (0 disables chunking).', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--use_quantum_mlp', help='Enable QuantumMLP for time embedding.', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--use_quantum_affine', help='Enable QuantumFrontEndQCNN (Quantum Affine) for spatial features.', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--quantum-frontendqcnn', help='Enable QuantumFrontEndQCNN frontend (decoupled from affine).', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--quantum-qcnn-chunk', help='Chunk size for QCNN batched simulation (0 means no chunking).', metavar='INT', type=click.IntRange(min=0), default=16384, show_default=True)
@click.option('--quantum-qcnn-strided', help='Enable strided CNOT entanglement in QCNN.', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--quantum-qcnn-reupload', help='Enable data re-uploading between QCNN layers.', metavar='BOOL', type=bool, default=False, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    c.real_p = opts.real_p
    c.train_on_latents = opts.train_on_latents
    c.progressive = opts.progressive

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    # [Quantum-Integration Marker] 模型构造位置（选择底层 UNet 架构）
    # 说明：此处根据 --arch 设置 c.network_kwargs.model_type 及其相关超参数（SongUNet 或 DhariwalUNet）。
    # 注意：真正的底层 UNet 实例化发生在 training/networks.py 的预条件化类（*Precond）__init__ 中，
    # 通过 self.model = globals()[model_type](...) 完成实例化。
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'quantum_transformer':
        # Quantum Transformer: use dedicated denoiser backbone with QSANN attention.
        c.network_kwargs.update(model_type='QuantumTransformerDenoiser')
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    # [Quantum-Integration Marker] 模型构造位置（选择预条件化封装器）
    # 说明：此处设置 c.network_kwargs.class_name 为具体预条件化类（VP/VE/EDM/Patch_EDM）。
    # 训练循环将据此在对应的 training.networks.*Precond 类中构造 self.model（底层 UNet）。
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    elif opts.precond == 'pedm':
        c.network_kwargs.class_name = 'training.networks.Patch_EDMPrecond'
        c.loss_kwargs.class_name = 'training.patch_loss.Patch_EDMLoss'
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
        # c.augment_kwargs.update(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        # c.network_kwargs.augment_dim = 6
    if opts.implicit_mlp:
        c.network_kwargs.implicit_mlp = True
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)
    # Quantum Transformer specific kwargs
    # Device & gradient clipping.
    if opts.device == 'cuda' and not torch.cuda.is_available():
        c.device = 'cpu'
    else:
        c.device = opts.device

    if opts.grad_clip_norm is not None:
        c.grad_clip_norm = float(opts.grad_clip_norm)

    if opts.arch == 'quantum_transformer':
        c.network_kwargs.update(model_dim=opts.model_dim,
                                num_heads=opts.heads,
                                layers=opts.layers,
                                patch_size=opts.patch_size,
                                quantum_n_qubits=opts.quantum_n_qubits,
                                quantum_q_depth=opts.quantum_depth,
                                quantum_qk_dim=opts.qk_dim,
                                quantum_attn_dropout=opts.quantum_attn_dropout,
                                quantum_attn_gate_init=opts.quantum_attn_gate_init,
                                force_fp32_attention=opts.force_fp32_attn)
        # Pass attention type choice to transformer denoiser
        c.network_kwargs.update(attn_type=opts.attn_type)

    # Enable quantum attention inside classical UNet blocks when requested.
    # This only applies to ddpmpp/ncsnpp/adm architectures.
    if opts.arch in ('ddpmpp', 'ncsnpp', 'adm') and opts.quantum_attn_in_unet:
        # Switch on quantum attention path in UNetBlock via SongUNet/DhariwalUNet constructor.
        c.network_kwargs.update(use_quantum_transformer=True)
        # Adapter import path (module:Class) and its kwargs.
        adapter_path = opts.quantum_adapter
        adapter_kwargs = dict(
            N_QUBITS=opts.quantum_n_qubits,
            Q_DEPTH=opts.quantum_depth,
            qk_dim=opts.qk_dim,
            encoding=opts.quantum_encoding,
            attn_dropout=opts.quantum_attn_dropout,
            qk_norm=opts.quantum_qk_norm,
            prefer_x_interface=True,
            force_fp32_attention=opts.force_fp32_attn,
            attn_chunk_size=opts.quantum_attn_chunk,
        )
        if opts.quantum_tau is not None:
            adapter_kwargs['tau'] = float(opts.quantum_tau)
        adapter_kwargs['tau_trainable'] = bool(opts.quantum_tau_trainable)
        c.network_kwargs.update(quantum_adapter=adapter_path, quantum_adapter_kwargs=adapter_kwargs)
        
    # Pass QuantumMLP and QuantumFrontEndQCNN flags to network_kwargs
    # Default: enabling quantum MLP also enables quantum affine
    if opts.use_quantum_mlp:
        opts.use_quantum_affine = True
    c.network_kwargs.update(use_quantum_mlp=opts.use_quantum_mlp)
    c.network_kwargs.update(use_quantum_affine=opts.use_quantum_affine)
    c.network_kwargs.update(use_qcnn_frontend=opts.quantum_frontendqcnn)
    c.network_kwargs.update(qcnn_chunk_size=opts.quantum_qcnn_chunk)
    c.network_kwargs.update(qcnn_use_strided=opts.quantum_qcnn_strided)
    c.network_kwargs.update(qcnn_reupload=opts.quantum_qcnn_reupload)

    # No legacy adapter path: quantum options are active only when arch=quantum_transformer.

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cpu'))
        if dist.get_world_size() > 1:
            seed = seed.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)
    # [Quantum-Integration Marker] 实例化触发点
    # 说明：training_loop.training_loop(**c) 将读取 c.network_kwargs.class_name 和 model_type，
    # 在 training/networks.py 的预条件化类 __init__ 中执行 self.model = globals()[model_type](...)，
    # 完成底层 UNet 模型的实例化；UNetBlock 内 attention=True 的位置将启用自注意力（可替换为量子 Transformer）。

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
