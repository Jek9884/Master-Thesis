from image_model import grid_search
import numpy as np
import ray
from ray import tune
import os
import torch
import random
import argparse

parser = argparse.ArgumentParser(description="Train the image autoencoder")

# Define command-line arguments
parser.add_argument("--file_path", type=str, help="Path to the dataset")
parser.add_argument("--save_path", type = str, help="Path to save results file")
# Parse the command-line arguments
args = parser.parse_args()
file_path = args.file_path
save_path = args.save_path
directory = "../../../Master-Thesis"
save_path = f"{directory}/{save_path}"

#Init raytune
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device(0)

ray.shutdown()
ray.init(num_gpus=1)

#file_path = "./dataset_{n_ckpt}_{k}_{cut}.pt"
images_tensor = torch.load(file_path)
random.seed(42)

indices = np.arange(len(images_tensor))
np.random.shuffle(indices)

split_idx = int(0.8 * len(images_tensor))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]
train_images, val_images = images_tensor[train_idx], images_tensor[val_idx]

params = {
    'batch_size' : tune.grid_search([8, 16, 32]),
    'lr' : tune.grid_search([1e-3, 1e-5, 1e-7]),
    'eps' : tune.grid_search([1e-8]),
    'weight_decay' : tune.grid_search([1e-3, 1e-5]),
    'epochs' : tune.grid_search([50, 100]),
    'embedding_dim' : tune.grid_search([128, 256]),
    'n_channels' : tune.grid_search([1]) 
}

train_images = train_images[:, np.newaxis, :, :]
val_images = val_images[:, np.newaxis, :, :]

grid_search(train_images, val_images, params, device, save_path)
