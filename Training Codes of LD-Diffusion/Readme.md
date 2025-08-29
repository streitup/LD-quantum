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
torchrun --standalone --nproc_per_node=1 train.py --outdir=training-runs --data=100-shot-obama.zip --cond=0 --arch=ncsnpp --batch=64 --batch-gpu=64 --augment=0.1 --real_p=1.0 --dropout=0.1 --fp16=True --xflip=True --ls=100 --train_on_latents=1
```

# Important notes

1. The codes of this module is built upon the codes of the Patch Diffusion [[link]](https://github.com/Zhendong-Wang/Patch-Diffusion), EDM [[link]](https://github.com/NVlabs/edm), DiffAug-GAN [[link]](https://github.com/mit-han-lab/data-efficient-gans) and ADA [[link]](https://github.com/NVlabs/stylegan2-ada-pytorch). We thank them a lot for their great work.

2. We have also found that certain hyperparameters during both training and sampling processes — even those adopted from previous work — as well as inherent properties of the dataset, can influence the performance of LD-Diffusion. If you wish to train or apply LD-Diffusion on your own dataset, we recommend tuning the hyperparameters specifically for your dataset to achieve optimal performance.

3.It will take about 4 days on one 4090 GPU with 40000kimg iteration on low-shot datasets, and the checkpoint is saved in training-runs/.

4. You should select the checkpoint to use the provided evaluation module to choose the best checkpoint. The result of the pre-trained LD-Diffusion model you selected should be close to the FID value reported in the paper.

5. Feel free to contact me at zzhang55@qub.ac.uk if you have any questions.

# Citation:

```
@inproceedings{zhang2025training,
  title={Training diffusion-based generative models with limited data},
  author={Zhang, Zhaoyu and Hua, Yang and Sun, Guanxiong and Wang, Hui and McLoone, Se{\'a}n},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
