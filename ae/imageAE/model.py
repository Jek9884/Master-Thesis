import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
import numpy as np
import random
import wandb

class ImageAutoencoder(LightningModule):

    def __init__(self, n_channels, height, width, latent_dim, loss_fn=F.mse_loss, lr=1e-3, eps=1e-8, weight_decay=0, seed=42):
        super(ImageAutoencoder, self).__init__()

        self.save_hyperparameters()

        self.conv_kernel = 4
        self.pool_kernel = 2
        self.stride = 2
        self.padding = 1

        self.total_train_loss = 0
        self.num_train_batches = 0

        self.total_val_loss = 0
        self.num_val_batches = 0

        #First Conv2D
        self.height_first_conv = ((height - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #42
        self.width_first_conv =  ((width - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #42
        print(f"After first conv: {self.height_first_conv} - {self.width_first_conv}")
        #First MaxPool2D
        self.height_first_pool = ((self.height_first_conv - self.pool_kernel) // self.stride) + 1 #21
        self.width_first_pool =  ((self.width_first_conv - self.pool_kernel) // self.stride) + 1 #21
        print(f"After first pool: {self.height_first_pool} - {self.width_first_pool}")
        #Second Conv2D
        self.height_second_conv = ((self.height_first_pool - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #10
        self.width_second_conv =  ((self.width_first_pool - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #10
        print(f"After second conv: {self.height_second_conv} - {self.width_second_conv}")
        #Second MaxPool2D
        self.height_second_pool = ((self.height_second_conv - self.pool_kernel) // self.stride) + 1 #5
        self.width_second_pool =  ((self.width_second_conv - self.pool_kernel) // self.stride) + 1 #5
        print(f"After second pool: {self.height_second_pool} - {self.width_second_pool}")
        #Third Conv2D
        #self.height_third_conv = ((self.height_second_pool - self.conv_kernel + (2*self.padding)) // 1) + 1 #4
        #self.width_third_conv =  ((self.width_second_pool - self.conv_kernel + (2*self.padding)) // 1) + 1 #4
        #print(f"After third conv: {self.height_third_conv} - {self.width_third_conv}")
        #Fourth Conv2D
        #self.height_fourth_conv = ((self.height_third_conv - self.conv_kernel + (2*self.padding)) // 1) + 1 #3
        #self.width_fourth_conv =  ((self.width_third_conv - self.conv_kernel + (2*self.padding)) // 1) + 1 #3
        #print(f"After fourth conv: {self.height_fourth_conv} - {self.width_fourth_conv}")
        #Third MaxPool2D
        #self.height_third_pool = ((self.height_fourth_conv - self.pool_kernel) // self.stride) + 1 #1
        #self.width_third_pool =  ((self.width_fourth_conv - self.pool_kernel) // self.stride) + 1 #1
        #print(f"After third max pool: {self.height_third_pool} - {self.width_third_pool}")
        #Input dimension for the linear layer
        #self.final_dim = latent_dim * self.height_third_pool * self.width_third_pool #128
        self.final_dim = latent_dim * self.height_second_pool * self.width_second_pool #3200
        print(f"Final dim: {self.final_dim}")

        # Encoder
        self.encoder = nn.Sequential(
            # First conv
            nn.Conv2d(n_channels, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            # First pool
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            # Second conv
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            # Second pool
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            # Third conv
            #nn.Conv2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=1, padding=self.padding),
            #nn.ReLU(),
            # Fourth conv
            #nn.Conv2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=1, padding=self.padding),
            #nn.ReLU(),
            # Third pool
            #nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            nn.Flatten(),
            nn.Linear(self.final_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.final_dim),
            nn.ReLU(),
            nn.Unflatten(1, (latent_dim, self.height_second_pool, self.width_second_pool)), # Small CNN
            #nn.Unflatten(1, (latent_dim, self.height_third_pool, self.width_third_pool)), # Big CNN
            # Third pool
            #nn.Upsample(size=(self.height_fourth_conv, self.width_fourth_conv), mode='nearest'),
            # Fourth conv
            #nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=1, padding=self.padding),
            #nn.ReLU(),
            # Third conv
            #nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=1, padding=self.padding),
            #nn.ReLU(),
            # Second pool
            nn.Upsample(size=(self.height_second_conv, self.width_second_conv), mode='nearest'),
            # Second conv
            nn.ConvTranspose2d(latent_dim, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding, output_padding=self.padding),
            nn.ReLU(),
            # First pool
            nn.Upsample(size=(self.height_first_conv, self.width_first_conv), mode='nearest'),
            # First conv
            nn.ConvTranspose2d(latent_dim // 2, n_channels, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding, output_padding=0),
            nn.Sigmoid()
        )

        self.loss_fn = loss_fn

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.eps, weight_decay=self.hparams.weight_decay)

    def training_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.total_train_loss += loss
        self.num_train_batches += 1
        #self.log('train_loss', loss, on_step=False, on_epoch=True)
        #wandb.log({"train_loss": loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.total_val_loss += loss
        self.num_val_batches += 1
        #self.log('val_loss', loss, on_step=False, on_epoch=True)
        #wandb.log({"val_loss": loss.item()})

    def test_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        #self.log('test_loss', loss, on_step=False, on_epoch=True)
        #wandb.log({"test_loss": loss.item()})

    def on_train_epoch_end(self):
        # Calcola la media della loss
        avg_train_loss = self.total_train_loss / self.num_train_batches

        self.log("train_loss", avg_train_loss.item())
        print(f"Epoch [{self.current_epoch}/{self.trainer.max_epochs}], Train loss: {avg_train_loss}")
        #wandb.log({"train_loss": avg_train_loss.item()})

    def on_validation_epoch_end(self):
        # Calcola la media della loss
        avg_val_loss = self.total_val_loss / self.num_val_batches

        self.log("val_loss", avg_val_loss.item())
        print(f"Epoch [{self.current_epoch}/{self.trainer.max_epochs}], Val loss: {avg_val_loss}")
        #wandb.log({"val_loss": avg_val_loss.item()})