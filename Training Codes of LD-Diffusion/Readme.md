# Training Codes for ICML2025 paper Training Diffusion-based Generative Models with Limited Data

We provide the Training Codes for the LD-Diffusion. The evaluation moudle can be found in the evaluation parts.

## Requirement: 
Use the following commands with Miniconda3 to create and activate your edm Python environment.
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

## Training
To train your own LD-Diffusion model on 100-shot obama dataset, please run the following command on the one GPU:

```
torchrun --standalone --nproc_per_node=1 train.py --outdir=training-runs --data=100-shot-obama.zip --cond=0 --arch=ncsnpp --batch=64 --batch-gpu=64 --augment=0.1 --real_p=1.0 --dropout=0.1 --fp16=True --xflip=True --ls=100 --train_on_latents=1
```

It will takes about 4 days on one 4090 GPU about 40000kimg iteration and the checkpoint is saved in training-runs/.

You should select the checkpoint to use the provided evaluation module to choose the best checkpoint. The result of the pre-trained LD-Diffusion model you selected should be close to the FID value reported in the paper. 
