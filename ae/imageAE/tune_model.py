import numpy as np
import ray
from ray import tune, air
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import os
import torch
import random
import argparse
import wandb
import csv
from model import ImageAutoencoder
from torch import nn
import torch.optim as optim
from imageDataset import ImageDataset
from torch.utils.data import DataLoader
from train import train_autoencoder
from data_handler import normalize_images_tensor, scale_images_tensor, log_scale_images_tensor

def grid_search(train, val, params, device, file_path):
    
    params["train_pt"] = ray.put(train)
    params["val_pt"] = ray.put(val)
    params["device"] = device
    params["file_path"] = file_path

    # Set logs to be shown on the Command Line Interface every 30 seconds
    #reporter = tune.CLIReporter(max_report_frequency=30)

    # Starts grid search using RayTune
    tuner = tune.Tuner(tune.with_resources(trainable, {"cpu":2, "gpu":1}),
                        param_space = params,
                        tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                        #run_config = air.RunConfig(name="ImageAE", verbose=1, progress_reporter=reporter))
                        #run_config= air.RunConfig(callbacks=[WandbLoggerCallback(project="Master-Thesis")])
                        )

    
    tuner.fit()


def trainable(config_dict):

    config = {'batch_size' : config_dict['batch_size'], 
               'lr' : config_dict['lr'], 
               'eps' : config_dict['eps'], 
               'weight_decay' : config_dict['weight_decay'], 
               'epochs' : config_dict['epochs'], 
               'embedding_dim' : config_dict['embedding_dim'], 
               'criterion' : config_dict['criterion'], 
               'device' : config_dict['device'],
               'metric' : config_dict['metric']}
    
    wandb.init(project="Master-Thesis", group=config_dict['group_name'], config = config)
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

    file_path = config_dict['file_path']

    print(f"The current directory is: {os.getcwd()}")

    if os.path.exists(file_path):
        with open(file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';')
            config_res = [config_dict["lr"], config_dict['eps'], config_dict['weight_decay'], config_dict['batch_size'], config_dict['epochs'], config_dict['embedding_dim'], config_dict['n_channels'], last_train_loss, last_val_loss, train_loss_vec, val_loss_vec]
            csv_writer.writerow(config_res)
    else:
        with open(file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';')
            header = ['lr', 'eps', 'weight_decay', 'batch_size', 'epochs', 'embedding_dim', 'n_channels', 'last_train_loss', 'last_val_loss', 'train_loss_vec', 'val_loss_vec']
            csv_writer.writerow(header)
            config_res = [config_dict["lr"], config_dict['eps'], config_dict['weight_decay'], config_dict['batch_size'], config_dict['epochs'], config_dict['embedding_dim'], config_dict['n_channels'], last_train_loss, last_val_loss, train_loss_vec, val_loss_vec]
            csv_writer.writerow(config_res)

parser = argparse.ArgumentParser(description="Train the image autoencoder")

# Define command-line arguments
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--file_path", type=str, help="Path to the dataset")
parser.add_argument("--save_path", type = str, help="Path to save results file")
parser.add_argument("--normalize", type=int, help="Normalize data before computation")
parser.add_argument("--log_scale", type=int, help="Log scale data before computation")

# Parse the command-line arguments
args = parser.parse_args()
dev = args.device
file_path = args.file_path
save_path = args.save_path
normalize = args.normalize
log_scale = args.log_scale

directory = "../../../Master-Thesis"
save_path = f"{directory}/{save_path}"

# Init raytune
os.environ["CUDA_VISIBLE_DEVICES"] = dev
device = torch.device(0)

# Init wandb logger
#wandb.init(project="Master-Thesis")

ray.shutdown()
ray.init(num_gpus=1)

images_tensor = torch.load(file_path)

images_tensor = images_tensor[:10000, :, :, :]

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

params = {
    'batch_size' : tune.grid_search([8, 16, 32]),
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
    'group_name' : 'Alien_10000_small'
}

grid_search(train_images, val_images, params, device, save_path)

ray.shutdown()
wandb.finish()

