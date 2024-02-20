import torch
from concurrent.futures import ThreadPoolExecutor
from data_handler import get_episodes
import argparse

parser = argparse.ArgumentParser(description="Extract images dataset from the experience replay dataset")

# Define command-line arguments
parser.add_argument("--n_seed", type=int, help="Number of seeds to use")
parser.add_argument("--n_ckpt", type=int, help="Number of checkpoints to use")
parser.add_argument("--k", type=int, help="How many samples for each checkpoint")

# Parse the command-line arguments
args = parser.parse_args()

# Load dataset 
directory = "../../dataset"

games = ["AirRaid", "Alien", "IceHockey", "Pong", "SpaceInvaders"]
n_seed = args.n_seed
n_ckpt = args.n_ckpt
k = args.k
cut = 100

images_tensor = torch.empty(0, dtype=torch.float32)


def process_checkpoint(game_path, seed, ckpt, k, cut):
    df = get_episodes(game_path, seed, ckpt, k, cut)
    images_list = df['Observation'].tolist()
    images_tensor = torch.tensor(images_list)
    return images_tensor

for seed in range(1,n_seed+1):

    with ThreadPoolExecutor() as executor:
        params = [(f"{directory}/{game}", seed, ckpt, k, cut) for game in games for ckpt in range(n_ckpt)]
        
        # Submit tasks to executor
        results = [executor.submit(process_checkpoint, *param) for param in params]

final_images_tensor = torch.cat([future.result() for future in results], dim=0)
print(final_images_tensor.shape)

file_path = f"./dataset_{n_ckpt}_{n_seed}_{k}_{cut}.pt"
torch.save(final_images_tensor, file_path)