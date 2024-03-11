import argparse
import torch
import os
import random
import numpy as np
from data_handler import log_scale_images_tensor
from train import train_autoencoder
from model import ImageAutoencoder
from imageDataset import ImageDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn

parser = argparse.ArgumentParser(description="Train the image autoencoder")

# Define command-line arguments

# Data related
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--file_path", type=str, help="Path to the dataset")
parser.add_argument("--save_path", type=str, help="Name to save the file")
parser.add_argument("--normalize", type=int, help="Normalize data before computation")
parser.add_argument("--log_scale", type=int, help="Log scale data before computation")
parser.add_argument("--n_channels", type=int, help="Number of the channels of each example")
parser.add_argument("--height", type=int, help="Height of each example")
parser.add_argument("--width", type=int, help="Width of each example")

# Hyper-params
parser.add_argument("--n_epochs", type=int, help="Number of epochs to perform")
parser.add_argument("--embedding_dim", type=int, help="Size of the latent space")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--criterion", type=str, help="Loss function")
parser.add_argument("--metric", type=str, help="Metric function")
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--eps", type=float, help="Adam precision")
parser.add_argument("--weight_decay", type=float, help="Weight decay")



# Parse the command-line arguments
# Data related
args = parser.parse_args()
dev = args.device
file_path = args.file_path
save_path = args.save_path
normalize = args.normalize
log_scale = args.log_scale
n_channels = args.n_channels
height = args.height
width = args.width

# Hyper-params
n_epochs = args.n_epochs
embedding_dim = args.embedding_dim
batch_size = args.batch_size
criterion = args.criterion
metric = args.metric
lr = args.lr
eps = args.eps
weight_decay = args.weight_decay

directory = "../../../Master-Thesis"

# Select device
os.environ["CUDA_VISIBLE_DEVICES"] = dev
device = torch.device(0)

# Load data
images_tensor = torch.load(file_path)
images_tensor = images_tensor[:10000, :, :, :]

# Transform data
if normalize:
    images_tensor = images_tensor / 255.0
if log_scale:
    images_tensor = log_scale_images_tensor(images_tensor)

# Split data in train and validation set
random.seed(42)
indices = np.arange(len(images_tensor))
np.random.shuffle(indices)
split_idx = int(0.8 * len(images_tensor))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]
train_images, val_images = images_tensor[train_idx], images_tensor[val_idx]

# Load model and wrap data
model = ImageAutoencoder(n_channels, height, width, embedding_dim)
model.to(device)

train_dataset = ImageDataset(train_images)
val_dataset = ImageDataset(val_images)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Select loss
if criterion == 'mse':
    criterion = nn.MSELoss()
elif criterion == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()

# Select metric
if metric == 'mae':
    metric = nn.L1Loss()
else:
    metric = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), 
                        lr=lr,
                        eps = eps,
                        weight_decay = weight_decay)

train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, metric, device, n_epochs, log=False, save_path=save_path)