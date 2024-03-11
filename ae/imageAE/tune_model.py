import numpy as np
import ray
from ray import tune, air
from ray.air.integrations.wandb import WandbLoggerCallback
import os
import torch
import random
import argparse
import wandb
from model import ImageAutoencoder
from nature_model import NatureAE
from torch import nn
import torch.optim as optim
from imageDataset import ImageDataset
from torch.utils.data import DataLoader
from train import train_autoencoder
from data_handler import log_scale_images_tensor

def grid_search(train, val, params, device):
    
    params["train_pt"] = ray.put(train)
    params["val_pt"] = ray.put(val)
    params["device"] = device

    # Starts grid search using RayTune
    tuner = tune.Tuner(tune.with_resources(trainable, {"cpu":2, "gpu":1}),
                        param_space = params,
                        tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                        run_config= air.RunConfig(callbacks=[WandbLoggerCallback(project="Master-Thesis")])
                        )

    
    tuner.fit()


def trainable(config_dict):
    
    wandb.init(project="Master-Thesis")
    torch.cuda.empty_cache()

    model = ImageAutoencoder(config_dict['n_channels'], config_dict['height'], config_dict['width'], config_dict['embedding_dim'])
    model.to(config_dict['device'])

    train_data = ray.get(config_dict['train_pt'])
    val_data = ray.get(config_dict['val_pt'])
    train_dataset = ImageDataset(train_data)
    val_dataset = ImageDataset(val_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config_dict['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config_dict['batch_size'], shuffle=True, num_workers=4)

    if config_dict['criterion'] == 'mse':
        criterion = nn.MSELoss()
    elif config_dict['criterion'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    if config_dict['metric'] == 'mae':
        metric = nn.L1Loss()
    else:
        metric = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), 
                           lr=config_dict['lr'],
                           eps = config_dict['eps'],
                           weight_decay = config_dict['weight_decay'])
    
    last_train_loss, last_val_loss, train_loss_vec, val_loss_vec = train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, metric, config_dict['device'], config_dict['epochs'])

parser = argparse.ArgumentParser(description="Train the image autoencoder")

# Define command-line arguments
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--file_path", type=str, help="Path to the dataset")
parser.add_argument("--normalize", type=int, help="Normalize data before computation")
parser.add_argument("--log_scale", type=int, help="Log scale data before computation")

# Parse the command-line arguments
args = parser.parse_args()
dev = args.device
file_path = args.file_path
normalize = args.normalize
log_scale = args.log_scale

directory = "../../../Master-Thesis"

os.environ["CUDA_VISIBLE_DEVICES"] = dev
device = torch.device(0)

# Init raytune
ray.shutdown()
ray.init(num_gpus=1)

images_tensor = torch.load(file_path)

#images_tensor = images_tensor[:10000, :, :, :]

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

# Grid search parameters
params = {
    'batch_size' : tune.grid_search([32]),
    'lr' : tune.grid_search([1e-3, 1e-5, 1e-7]),
    'eps' : tune.grid_search([1e-8]),
    'weight_decay' : tune.grid_search([1e-1, 1e-3, 1e-5]),
    'epochs' : tune.grid_search([300]),
    'embedding_dim' : tune.grid_search([128, 256, 512]),
    'n_channels' : tune.grid_search([4]),
    'height' : tune.grid_search([84]),
    'width' : tune.grid_search([84]),
    'criterion' : tune.grid_search(['mse']),
    'metric' : tune.grid_search(['mse']),
    'game' : tune.grid_search(["Alien"]),
    'n_samples' : tune.grid_search(["56833"]),
    'model_type' : tune.grid_search(["small"]),
    'normalize' : tune.grid_search([normalize]),
    'log_scale' : tune.grid_search([log_scale])
}

grid_search(train_images, val_images, params, device)

ray.shutdown()
wandb.finish()

