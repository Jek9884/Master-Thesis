import torch
import argparse
import os
import random
import numpy as np
import torch.optim as optim
from imageDataset import ImageDataset
from torch.utils.data import DataLoader
from model import ImageAutoencoder
from data_handler import log_scale_images_tensor
from torch import nn


# Define the training function
def test_autoencoder(model, test_dataloader, criterion, metric, device):

    with torch.no_grad():
        for test_data in test_dataloader:
            test_images = test_data
            for image in test_images:
                image = image.unsqueeze(0)
            test_images = test_images.to(device)

            #Forward pass
            test_outputs = model(test_images)

            # Compute the validation loss
            test_loss = criterion(test_outputs, test_images)

            # Compute the validation metric
            test_metric = metric(test_outputs, test_images)

            total_test_metric += test_metric.item()
            total_test_loss += test_loss.item()
    
    # Print average loss and MSE for the epoch
    average_test_loss = total_test_loss / len(test_dataloader)
    average_test_metric = total_test_metric / len(test_dataloader)

    return average_test_loss, average_test_metric

parser = argparse.ArgumentParser(description="Test the image autoencoder")

# Define command-line arguments

# Data related
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--model_path", type=str, help="Path to the model")
parser.add_argument("--file_path", type=str, help="Path to the dataset")
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
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--eps", type=float, help="Optimizer precision")
parser.add_argument("--weight_decay", type=float, help="Weight decay")

# Parse the command-line arguments
args = parser.parse_args()

# Data related
dev = args.device
model_path = args.model_path
file_path = args.file_path
normalize = args.normalize
log_scale = args.log_scale
n_channels = args.n_channels
height = args.height
width = args.widht

# Hyper-params
latent_dim = args.latent_dim
batch_size = args.batch_size
criterion = args.criterion
metric = args.metric
lr = args.lr
eps = args.eps
weight_decay = args.weight_decay

directory = "../../../Master-Thesis"

os.environ["CUDA_VISIBLE_DEVICES"] = dev
device = torch.device(0)

images_tensor = torch.load(file_path)

if normalize:
    images_tensor = images_tensor / 255.0
if log_scale:
    images_tensor = log_scale_images_tensor(images_tensor)

model = ImageAutoencoder(n_channels, height, width, latent_dim)
model.to(device)

test_dataset = ImageDataset(images_tensor)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

if criterion == 'mse':
    criterion = nn.MSELoss()
elif criterion == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()

if metric == 'mae':
    metric = nn.L1Loss()
else:
    metric = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), 
                        lr=lr,
                        eps = eps,
                        weight_decay = weight_decay)

avg_test_loss, avg_test_metric = test_autoencoder(model, test_dataloader, criterion, metric, device)

print(f"Test avg loss: {avg_test_loss}")
print(f"Test avg metric: {avg_test_metric}")