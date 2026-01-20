# Training Codes for ICML2025 paper Training Diffusion-based Generative Models with Limited Data

We provide the Training Codes for the LD-Diffusion. The evaluation module can be found in the evaluation parts.

# Dataset

The low-shot datasets can be found in [[link]](https://drive.google.com/file/d/1rWqaVlms55604jrP5t9ShacL6mZKWL8f/view?usp=sharing).

# Requirement: 
Use the following commands with Miniconda3 to create and activate your EDM Python environment.
```
conda env create -f environment.yml
```


```
conda activate edm
```

```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

```
conda install diffusers
```

# Training
To train your own LD-Diffusion model on the 100-shot Obama dataset, please run the following command on one GPU:

```
torchrun --standalone --nproc_per_node=1 train.py \
  --outdir=training-runs \
  --data=100-shot-obama.zip \
  --cond=0 \
  --arch=ncsnpp \
  --batch=64 --batch-gpu=64 \
  --augment=0.1 \
  --real_p=1.0 \
  --dropout=0.1 \
  --fp16=True \
  --xflip=True \
  --ls=100 \
  --train_on_latents=1
```

Notes:
- Default preconditioning is now set to `edm` (original EDMLoss, predicting clean target x0). You can omit `--precond` and it will use EDM by default.
- If you want patch-based training, explicitly set `--precond=pedm`. In EDM mode, no extra position channels or patch-specific args are used.
- When `--train_on_latents=1`, training uses diffusers `AutoencoderKL` (stabilityai/sd-vae-ft-mse) to encode images to 4-channel latents scaled by 0.18215.
 - Default architecture is now `ncsnpp`. To enable the new Quantum Transformer model, pass `--arch=quantum_transformer` (the quantum attention path is opt-in and otherwise disabled).

Quantum Transformer (QSANN) quick start:
- Requires torchquantum. Install: `pip install torchquantum`.
- Architecture must be set to `quantum_transformer`. Preconditioning can be `edm` (recommended for latent-space) or `pedm`.

Minimal training example (EDM, latent 4-channels, QSANN attention):

```
torchrun --standalone --nproc_per_node=1 train.py \
  --outdir=training-runs \
  --data=100-shot-obama.zip \
  --cond=0 \
  --arch=quantum_transformer \
  --precond=edm \
  --batch=64 --batch-gpu=64 \
  --augment=0.1 \
  --real_p=1.0 \
  --dropout=0.0 \
  --fp16=True \
  --xflip=True \
  --ls=100 \
  --train_on_latents=1 \
  --model-dim=384 --heads=8 --layers=4 --patch-size=4 \
  --quantum-n-qubits=6 --quantum-depth=2 --qk-dim=4 \
  --quantum-attn-dropout=0.1 --quantum-attn-gate-init=0.5 \
  --force-fp32-attn=1
```

Notes for QSANN parameters:
- `--patch-size=4` implies 32×32 latent → L=64 tokens at resolution 128 (or 16 tokens at 64 depending on latent resolution); ensure H,W divisible by p.
- `--quantum-n-qubits=6` is fixed for 64-d amplitude encoding.
- `--qk-dim=4` sets the Q/K projection dimension inside QuantumAttention64.
- `--force-fp32-attn=1` keeps attention ops in FP32 under AMP to avoid numerical issues.

Docs:
- Quantum Transformer plan and integration details: `docs/quantum_transformer_unet_plan.md`
- Adapter and integration notes: `quantum_adapt.md`

# Important notes

1. The codes of this module is built upon the codes of the Patch Diffusion [[link]](https://github.com/Zhendong-Wang/Patch-Diffusion), EDM [[link]](https://github.com/NVlabs/edm), DiffAug-GAN [[link]](https://github.com/mit-han-lab/data-efficient-gans) and ADA [[link]](https://github.com/NVlabs/stylegan2-ada-pytorch). We thank them a lot for their great work.

2. We have also found that certain hyperparameters during both training and sampling processes — even those adopted from previous work — as well as inherent properties of the dataset, can influence the performance of LD-Diffusion. If you wish to train or apply LD-Diffusion on your own dataset, we recommend tuning the hyperparameters specifically for your dataset to achieve optimal performance.

3. It will take about 4 days on one 4090 GPU with 40000kimg iteration on low-shot datasets, and the checkpoint is saved in training-runs/.

4. You should select the checkpoint to use the provided evaluation module to choose the best checkpoint. The result of the pre-trained LD-Diffusion model you selected should be close to the FID value reported in the paper.

5. Feel free to contact me at zzhang55@qub.ac.uk if you have any questions.

# Sampling

We provide a unified `generate.py` script. For latent-space EDM sampling, add `--on_latents=1`; the script will sample 4-channel latents at 32×32 and decode them via `AutoencoderKL.decode` back to 128×128 RGB images.

Example (single GPU):

```
python generate.py \
  --network=training-runs/<RUN_DIR>/network-snapshot-<KIMG>.pkl \
  --outdir=generated_images \
  --seeds=0-63 \
  --batch=64 \
  --on_latents=1 \
  --resolution=64
```

Tips:
- For EDM latent sampling, `generate.py` sets latent image_size=32, resolution=32 and channel=4 internally.
- If your model was trained with `--precond=pedm` (Patch-EDM), do not use `--on_latents`; sample as normal images and keep positional channels masked by `--mask_pos=True` if needed.

Further docs:
- Refactor plan (EDM default + latent training): docs/改造方案.md
- Quantum Transformer replaces UNet (EDM + latent 16×16): docs/quantum_transformer_unet_plan.md

Migration note (arch):
- Older runs may have enabled quantum attention directly via flags. In the current version, the Quantum Transformer is only activated when you set `--arch=quantum_transformer`. All other `arch` values (`ncsnpp`, `ddpmpp`, `adm`) will keep classical attention paths, ignoring quantum-attention toggles.

# Minimal training + sampling combo (EDM latent space)

Training (EDM, latent space, single GPU):

```
torchrun --standalone --nproc_per_node=1 train.py \
  --outdir=training-runs \
  --data=100-shot-obama.zip \
  --cond=0 \
  --arch=ncsnpp \
  --batch=64 --batch-gpu=64 \
  --augment=0.1 \
  --real_p=1.0 \
  --dropout=0.1 \
  --fp16=True \
  --xflip=True \
  --ls=100 \
  --train_on_latents=1
```

Sampling (EDM, latent space, single GPU):

```
python generate.py \
  --network=training-runs/<RUN_DIR>/network-snapshot-<KIMG>.pkl \
  --outdir=generated_images \
  --seeds=0-63 \
  --batch=64 \
  --on_latents=1 \
  --resolution=64
```

# Citation:

```
@inproceedings{zhang2025training,
  title={Training diffusion-based generative models with limited data},
  author={Zhang, Zhaoyu and Hua, Yang and Sun, Guanxiong and Wang, Hui and McLoone, Se{\'a}n},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
