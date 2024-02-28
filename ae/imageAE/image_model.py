from torch import nn
import torch
from torch.utils.data import Dataset
from ray import tune
from ray import air
from torch.utils.data import DataLoader
import torch.optim as optim
import ray
import numpy as np
import os
import csv

# N == Input dimension
# K == Kernel size
# S == Stride
# P == Padding
# OP == Output padding
# The formula for output dimension after a convolution is ((N-K+2P) / S) + 1
# The formula for output dimension after a max pooling is ((N-K) / S) + 1
# The formula for output dimension after a deconvolution is ((N-1)*S)) - (2*P) + K + OP
class ImageAutoencoder(nn.Module):

    def __init__(self, n_channels, height, width, latent_dim):
        super(ImageAutoencoder, self).__init__()
        
        self.conv_kernel = 4
        self.pool_kernel = 2
        self.stride = 2
        self.padding = 1

        #First Conv2D
        self.height_final_dim = ((height - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        self.width_final_dim =  ((width - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        print(f'After first conv: {self.height_final_dim} - {self.width_final_dim}') #42
        #First MaxPool2D
        self.height_final_dim = ((self.height_final_dim - self.pool_kernel) // self.stride) + 1
        self.width_final_dim =  ((self.width_final_dim - self.pool_kernel) // self.stride) + 1
        print(f'After first max pool: {self.height_final_dim} - {self.width_final_dim}') #21
        #Second Conv2D
        self.height_final_dim = ((self.height_final_dim - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        self.width_final_dim =  ((self.width_final_dim - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        print(f'After second conv: {self.height_final_dim} - {self.width_final_dim}') #10
        #Second MaxPool2D
        self.height_final_dim = ((self.height_final_dim - self.pool_kernel) // self.stride) + 1
        self.width_final_dim =  ((self.width_final_dim - self.pool_kernel) // self.stride) + 1
        print(f'After second max pool: {self.height_final_dim} - {self.width_final_dim}') #5
        #Input dimension for the linear layer
        self.final_dim = latent_dim * self.height_final_dim * self.width_final_dim
        print(f'Final dim: {self.final_dim}') #3200

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            nn.Flatten(),
            nn.Linear(self.final_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.final_dim),
            nn.ReLU(),
            nn.Unflatten(1, (latent_dim, self.height_final_dim, self.width_final_dim)),
            nn.Upsample(scale_factor=self.pool_kernel, mode='nearest'),
            nn.ConvTranspose2d(latent_dim, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding, output_padding=self.padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=self.pool_kernel, mode='nearest'),
            nn.ConvTranspose2d(latent_dim // 2, n_channels, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding, output_padding=0),
            nn.ReLU()
        )

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the training function
def train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs = 10):

    train_loss_vec = []
    val_loss_vec = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for data in train_dataloader:
            images = data
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, images)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_loss_vec.append(loss.item())

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_data in val_dataloader:
                val_images = val_data
                for image in val_images:
                    image = image.unsqueeze(0)
                val_images = val_images.to(device)

                #Forward pass
                val_outputs = model(val_images)

                # Compute the validation loss
                val_loss = criterion(val_outputs, val_images)
                total_val_loss += val_loss.item()
                val_loss_vec.append(loss.item())

        
        # Print average loss for the epoch
        average_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}")
        
        average_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {average_val_loss:.4f}")
    
    return average_train_loss, average_val_loss, train_loss_vec, val_loss_vec
    

class CustomImageDataset(Dataset):
    def __init__(self, images_list, transform=None):
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = self.images_list[idx]

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32)

        return image
    
def grid_search(train, val, params, device, file_path):

    params["train_pt"] = ray.put(train)
    params["val_pt"] = ray.put(val)
    params["device"] = device
    params["file_path"] = file_path

    # Set logs to be shown on the Command Line Interface every 30 seconds
    reporter = tune.CLIReporter(max_report_frequency=30)

    # Starts grid search using RayTune
    tuner = tune.Tuner(tune.with_resources(trainable, {"cpu":2, "gpu":1}),
                        param_space = params,
                        tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                        run_config = air.RunConfig(name="ImageAE", verbose=1, progress_reporter=reporter))
    
    tuner.fit()

def trainable(config_dict):

    torch.cuda.empty_cache()

    model = ImageAutoencoder(config_dict['n_channels'], config_dict['height'], config_dict['width'], config_dict['embedding_dim'])
    model.to(config_dict['device'])

    train_data = ray.get(config_dict['train_pt'])
    val_data = ray.get(config_dict['val_pt'])
    train_dataset = CustomImageDataset(train_data)
    val_dataset = CustomImageDataset(val_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config_dict['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config_dict['batch_size'], shuffle=True, num_workers=4)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=config_dict['lr'],
                           eps = config_dict['eps'],
                           weight_decay = config_dict['weight_decay'])
    
    last_train_loss, last_val_loss, train_loss_vec, val_loss_vec = train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, config_dict['device'], config_dict['epochs'])

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