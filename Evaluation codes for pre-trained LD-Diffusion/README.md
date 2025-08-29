# Evaluation module and pre-trained model for Training Diffusion-based Generative Models with Limited Data

We provide the evaluation module and pre-trained LD-Diffusion model [[link]](https://drive.google.com/file/d/1yJsVgF4s8oh-eukOYebNSQZpGzlsEqvD/view?usp=sharing) on the 100-shot-obama dataset. This code is only used to evaluate the pre-trained LD-Diffusion models. 

## Requirement: 
Use the following commands with Miniconda3 to create and activate your edm Python environment.
```
conda env create -f environment.yml
```

```
conda activate edm
```

```
conda install diffusers
```


To evaluate the pre-trained LD-Diffusion model on the 100-shot-obama dataset, run the following command on two GPUs:

```
torchrun --standalone --nproc_per_node=2 generate.py --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003 --resolution 256 --on_latents=1 --batch 64 --outdir=fid-tmp --seeds=0-4999 --subdirs --network=LD-Diffusion_100-shot-obama.pkl
```

```
python fid.py ref --data=100-shot-obama.zip --dest=fid-refs/100-shot-obama.npz
```

```
torchrun --standalone --nproc_per_node=2 fid.py calc --images=fid-tmp --ref=fid-refs/100-shot-obama.npz
```

The evaluation result of the pre-trained LD-Diffusion model should be close to the FID value reported in the paper. Due to the large size of the pre-trained model, we only provide one pre-trained model on 100-shot-obama dataset here.

