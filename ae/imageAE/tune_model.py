import pytorch_lightning as pl
import argparse
import torch
import os
from ray import tune
import ray
import wandb
from ray.air.config import RunConfig
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from model import ImageAutoencoder
from imageDataset import ImageDataModule
from kfoldTrainer import KFoldTrainer

import sys
sys.path.insert(1, '/storagenfs/a.capurso1/Master-Thesis/ae/lightning_version/data_handler')
from utilities import normalize_images_tensor, log_scale_images_tensor

def trainable(config_dict):
    
    seed = 42
    pl.seed_everything(seed, workers=True)

    images_tensor = ray.get(config_dict['data_pt'])

    if config_dict['normalize']:
        images_tensor = normalize_images_tensor(images_tensor)
    if config_dict['log_scale']:
        images_tensor = log_scale_images_tensor(images_tensor)

    # Initialize Weights & Biases run
    #wandb.init(project="Master-Thesis", config=config_dict)

    # Define the model
    model = ImageAutoencoder(n_channels=config_dict['n_channels'], height=config_dict['height'], width=config_dict['width'], latent_dim=config_dict['embedding_dim'], lr=config_dict['lr'], weight_decay=config_dict['weight_decay'], eps=config_dict['eps'])

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=config_dict["patience"],
        mode="min"
    )

    wandb_logger = WandbLogger(project="Master-Thesis", config=config_dict)
    # Define a Lightning Trainer
    trainer = KFoldTrainer(max_epochs=config_dict['epochs'], num_folds=3, deterministic = True, enable_progress_bar = False, logger=wandb_logger, callbacks=[early_stopping_callback])
    trainer.fit(model, datamodule=ImageDataModule(images_tensor, batch_size=config_dict['batch_size'], num_workers=10))

    wandb.finish()

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

seed = 42
pl.seed_everything(seed, workers=True)

images_tensor = torch.load(file_path)
#images_tensor = images_tensor[:10000, :, :, :]

os.environ["CUDA_VISIBLE_DEVICES"] = dev
device = torch.device(0)

# Configure Ray Tune
ray.shutdown()
ray.init()

data_pt = ray.put(images_tensor)


# Define the search space
search_space = {
    'data_pt' : data_pt,
    'batch_size' : tune.grid_search([32, 64, 128]),
    'lr' : tune.grid_search([1e-3, 1e-5, 1e-7]),
    'eps' : tune.grid_search([1e-8]),
    'weight_decay' : tune.grid_search([1e-3, 1e-5, 1e-7]),
    'epochs' : tune.grid_search([500]),
    'embedding_dim' : tune.grid_search([128]),
    'n_channels' : tune.grid_search([4]),
    'height' : tune.grid_search([84]),
    'width' : tune.grid_search([84]),
    'criterion' : tune.grid_search(['mse']),
    'metric' : tune.grid_search(['mse']),
    'game' : tune.grid_search(["SpaceInvaders"]),
    'patience' : tune.grid_search([5, 10]),
    'divergence_threshold' : tune.grid_search([1e-6]),
    'n_samples' : tune.grid_search([images_tensor.shape[0]]),
    'model_type' : tune.grid_search(["small"]),
    'normalize' : tune.grid_search([1]),
    'log_scale' : tune.grid_search([0]),
    'seed' : tune.grid_search([seed])
}

tuner = tune.Tuner(tune.with_resources(trainable, 
                                       {"cpu":10,"gpu":0.5},
                                       ),
                    param_space = search_space,
                    run_config = RunConfig(name="game", verbose=1)
                )

results = tuner.fit()

ray.shutdown()