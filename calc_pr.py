
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Precision and Recall for Generative Models')
    parser.add_argument('--real_dir', type=str, required=True, help='Path to real images directory')
    parser.add_argument('--fake_dir', type=str, required=True, help='Path to generated images directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--k', type=int, default=3, help='Number of nearest neighbors for manifold estimation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    self.image_paths.append(os.path.join(root, file))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return torch.zeros(3, 299, 299) # Return dummy
        
        if self.transform:
            img = self.transform(img)
        return img

def get_inception_feature_extractor(device):
    # Use standard InceptionV3 pretrained on ImageNet
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity() # Remove classification head
    model.eval()
    model.to(device)
    return model

@torch.no_grad()
def extract_features(loader, model, device):
    features_list = []
    pbar = tqdm(loader, desc="Extracting features")
    for batch in pbar:
        batch = batch.to(device)
        # Inception expects 299x299 inputs normalized to [-1, 1] usually? 
        # But torchvision model expects normalized with mean/std.
        # However, standard FID/PR uses [0, 1] range inputs?
        # Actually torchvision Inception v3 expects:
        # Resize to 299, CenterCrop 299, ToTensor, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # We handle this in transform.
        
        # Output is (N, 2048)
        feats = model(batch)
        features_list.append(feats.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    return features

def compute_manifold(features, k=3):
    """
    Compute the k-th nearest neighbor distance for each feature vector.
    """
    print(f"Computing k-NN (k={k})...")
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='euclidean').fit(features)
    distances, _ = nbrs.kneighbors(features)
    # The 0-th neighbor is the point itself, so we take the k-th neighbor (index k)
    radii = distances[:, k]
    return radii, nbrs

def compute_precision_recall(real_features, fake_features, k=3):
    """
    Calculate Precision and Recall.
    Precision: How many fake features are within the manifold of real features.
    Recall: How many real features are within the manifold of fake features.
    """
    # 1. Manifold of Real
    real_radii, real_nbrs = compute_manifold(real_features, k)
    
    # Check overlap of Fake in Real Manifold (Precision)
    print("Computing Precision...")
    # Find nearest real neighbor for each fake sample
    distances_fake_to_real, indices_fake_to_real = real_nbrs.kneighbors(fake_features, n_neighbors=1)
    distances_fake_to_real = distances_fake_to_real[:, 0]
    nearest_real_indices = indices_fake_to_real[:, 0]
    
    # Condition: distance <= radius of the nearest real neighbor
    precision_mask = distances_fake_to_real <= real_radii[nearest_real_indices]
    precision = np.mean(precision_mask)
    
    # 2. Manifold of Fake
    fake_radii, fake_nbrs = compute_manifold(fake_features, k)
    
    # Check overlap of Real in Fake Manifold (Recall)
    print("Computing Recall...")
    distances_real_to_fake, indices_real_to_fake = fake_nbrs.kneighbors(real_features, n_neighbors=1)
    distances_real_to_fake = distances_real_to_fake[:, 0]
    nearest_fake_indices = indices_real_to_fake[:, 0]
    
    recall_mask = distances_real_to_fake <= fake_radii[nearest_fake_indices]
    recall = np.mean(recall_mask)
    
    return precision, recall

def main():
    args = parse_args()
    
    # Transform for InceptionV3
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print(f"Loading Real images from {args.real_dir}")
    real_dataset = ImageFolderDataset(args.real_dir, transform=transform)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Loading Fake images from {args.fake_dir}")
    fake_dataset = ImageFolderDataset(args.fake_dir, transform=transform)
    fake_loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Model
    model = get_inception_feature_extractor(args.device)
    
    # Extract
    real_features = extract_features(real_loader, model, args.device)
    fake_features = extract_features(fake_loader, model, args.device)
    
    print(f"Real Features: {real_features.shape}")
    print(f"Fake Features: {fake_features.shape}")
    
    # Calculate PR
    precision, recall = compute_precision_recall(real_features, fake_features, k=args.k)
    
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("-" * 30)
    
    # Save results
    with open("pr_results.txt", "w") as f:
        f.write(f"Real: {args.real_dir}\n")
        f.write(f"Fake: {args.fake_dir}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
    print("Results saved to pr_results.txt")

if __name__ == "__main__":
    main()
