import argparse
import os
import torch
from imageDataset import ImageDataset
from torch.utils.data import DataLoader
import numpy as np

import sys
sys.path.insert(1, '..')
from model import ImageAutoencoder
import sys
sys.path.insert(1, 'data_handler')
from data_handler.data_handler import normalize_images_tensor, log_scale_images_tensor

parser = argparse.ArgumentParser(description="Test the image autoencoder")

# Data related
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--model_path", type=str, help="Path to the model")
parser.add_argument("--file_path", type=str, help="Path to the dataset")
parser.add_argument("--save_path", type=str, help="Path where to save embeddings")
parser.add_argument("--normalize", type=int, help="Normalize data before computation")
parser.add_argument("--log_scale", type=int, help="Log scale data before computation")
parser.add_argument("--n_channels", type=int, help="Number of channels for each example")
parser.add_argument("--height", type=int, help="Height of each image")
parser.add_argument("--widht", type=int, help="Width of each image")

# Hyper-params
parser.add_argument("--latent_dim", type=int, help="Latent dimension")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--criterion", type=str, help="Loss function")
parser.add_argument("--metric", type=str, help="Metric to use")

# Parse the command-line arguments
args = parser.parse_args()

# Data related
dev = args.device
model_path = args.model_path
file_path = args.file_path
save_path = args.save_path
normalize = args.normalize
log_scale = args.log_scale
n_channels = args.n_channels
height = args.height
width = args.widht

# Hyper-params
latent_dim = args.latent_dim

directory = "../../../Master-Thesis"

os.environ["CUDA_VISIBLE_DEVICES"] = dev
device = torch.device(0)

images_tensor = torch.load(file_path)

if normalize:
    images_tensor = normalize_images_tensor(images_tensor)
if log_scale:
    images_tensor = log_scale_images_tensor(images_tensor)

model = ImageAutoencoder(n_channels, height, width, latent_dim)
model.to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)

images_dataset = ImageDataset(images_tensor)
images_dataloader = DataLoader(images_dataset, batch_size=1, shuffle=True, num_workers=4)

with torch.no_grad():

    embeddings_list = None

    model.eval()

    for data in images_dataloader:
        images = data
        for image in images:
            image = image.unsqueeze(0)
        images = images.to(device)

        #Forward pass
        embeddings = model(images, return_encodings=True)

        embeddings = embeddings.cpu()

        if embeddings_list is None:
            embeddings_list = embeddings
        else:
            embeddings_list = torch.cat((embeddings_list, embeddings), dim=0)

torch.save(embeddings_list, save_path)