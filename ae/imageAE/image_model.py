import numpy as np
import pandas as pd
import random
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np
import gzip
from concurrent.futures import ThreadPoolExecutor
import torch.optim as optim
import ray
from ray import tune
from ray import air
import os

def get_episodes_idx(dir_path, seed, ckpt):

    episodes_idx = []

    try:
        filename_term = f"{dir_path}/{seed}/replay_logs/$store$_terminal_ckpt.{ckpt}.gz"
        pickled_data = gzip.GzipFile(filename=filename_term)
        np_term = np.load(pickled_data)

        for idx, elem in enumerate(np_term):
            if elem == 1:
                episodes_idx.append(idx)

        return episodes_idx
    
    except FileNotFoundError:
        print(f"FIle {filename_term} not found.")
        return episodes_idx
    except PermissionError:
        print(f"Permission error: Unable to read {filename_term}.")
        return episodes_idx
    except Exception as e:
        print(f"An error occurred: {e}")
        return episodes_idx
    
def split_and_save_episodes(obs, act, rwd, idx_lst, k, cut):
    try:
        data = {
            'Observation': [],
            'Action': [],
            'Reward': []
        }

        for i in range(1, len(idx_lst)):
            start = idx_lst[i - 1] + 1
            end = idx_lst[i] + 1

            if start < len(obs) and end <= len(obs):
                # Randomly select an element along the 0th axis
                random_frame = random.randint(start, end - 1)

                data['Observation'].append(obs[random_frame])
                data['Action'].append(act[start:cut])
                data['Reward'].append(rwd[start:cut])

        df = pd.DataFrame(data)
        df['Rank'] = df.apply(lambda row: row['Reward'].sum(), axis=1)
        df = df.sort_values(by='Rank', ascending=False)
        df = df.head(k)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame({"Observation": [], "Action": [], "Reward": []})

def get_episodes(dir_path, seed, ckpt, k, cut):

    random.seed(42)

    idx_lst = get_episodes_idx(dir_path, seed, ckpt)
    
    if len(idx_lst) > 0:
        try:
            filename_obs = f"{dir_path}/{seed}/replay_logs/$store$_observation_ckpt.{ckpt}.gz"
            pickled_data = gzip.GzipFile(filename=filename_obs)
            obs = np.load(pickled_data)
            filename_act = f"{dir_path}/{seed}/replay_logs/$store$_action_ckpt.{ckpt}.gz"
            pickled_data = gzip.GzipFile(filename=filename_act)
            act = np.load(pickled_data)
            filename_rwd = f"{dir_path}/{seed}/replay_logs/$store$_reward_ckpt.{ckpt}.gz"
            pickled_data = gzip.GzipFile(filename=filename_rwd)
            rwd = np.load(pickled_data)

            df = split_and_save_episodes(obs, act, rwd, idx_lst, k, cut)

            return df
        
        except FileNotFoundError:
            print(f"Checkpoint {ckpt} not found.")
            return pd.DataFrame({"Observation": [], "Action": [], "Reward": []})
        except PermissionError:
            print(f"Permission error: Unable to read {ckpt}.")
            return pd.DataFrame({"Observation": [], "Action": [], "Reward": []})
        except Exception as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame({"Observation": [], "Action": [], "Reward": []})
        
    else:

        return pd.DataFrame({"Observation": [], "Action": [], "Reward": []})
    
# Define the ImageAutoencoder class
class ImageAutoencoder(nn.Module):
    def __init__(self, n_channels, encoded_size):
        super(ImageAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, encoded_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoded_size, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the training function
def train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=10):

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for data in train_dataloader:
            images = data
            for image in images:
                image = image.unsqueeze(0)
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

        
        # Print average loss for the epoch
        average_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}")
        
        average_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {average_val_loss:.4f}")
        

    print("Training completed!")

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

        # Assuming images are 2D matrices, you may need to reshape them to (H, W, C)
        # and convert them to torch.Tensor
        image = torch.tensor(image, dtype=torch.float32)#.permute(2, 0, 1)

        return image
    
def grid_search(train, val, params, device):

    params["train_data"] = train
    params["val_data"] = val
    params["device"] = device

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

    model = ImageAutoencoder(config_dict['n_channels'], config_dict['embedding_dim'])
    model.to(config_dict['device'])

    train_dataset = CustomImageDataset(config_dict['train_data'])
    val_dataset = CustomImageDataset(config_dict['val_data'])

    train_dataloader = DataLoader(train_dataset, batch_size=config_dict['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config_dict['batch_size'], shuffle=True, num_workers=4)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=config_dict['lr'],
                           eps = config_dict['eps'],
                           weight_decay = config_dict['weight_decay'])
    
    train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, config_dict['epochs'])






   
#Init raytune
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device(0)

#ray.shutdown()
#ray.init(num_gpus=1)

# Load dataset 
directory = "dataset"
games = ["AirRaid", "Alien"]#, "IceHockey", "Pong", "SpaceInvaders"]
seed = 1
n_ckpt = 2
k = 2
cut = 100

images_tensor = torch.empty(0, dtype=torch.float32)


def process_checkpoint(game_path, seed, ckpt, k, cut):
    df = get_episodes(game_path, seed, ckpt, k, cut)
    images_list = df['Observation'].tolist()
    images_tensor = torch.tensor(images_list)
    return images_tensor

with ThreadPoolExecutor() as executor:
    params = [(f"{directory}/{game}", seed, ckpt, k, cut) for game in games for ckpt in range(n_ckpt)]
    
    # Submit tasks to executor
    results = [executor.submit(process_checkpoint, *param) for param in params]

final_images_tensor = torch.cat([future.result() for future in results], dim=0)
print(final_images_tensor.shape)

random.seed(42)

indices = np.arange(len(final_images_tensor))
np.random.shuffle(indices)

split_idx = int(0.8 * len(final_images_tensor))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]
train_images, val_images = final_images_tensor[train_idx], final_images_tensor[val_idx]

"""
params = {
    'batch_size' : 8,
    'lr' : 1e-3,
    'eps' : 1e-8,
    'weight_decay' : 1e-8,
    'epochs' : 1,
    'embedding_dim' : 128,
    'n_channels' : 1 
}

grid_search(train_images, val_images, params, device)"""

train_dataset = CustomImageDataset(train_images)
val_dataset = CustomImageDataset(val_images)

batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = ImageAutoencoder(1, 128)

model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, device, 1)