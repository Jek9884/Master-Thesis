import argparse
from lightning import seed_everything
from data_extraction import get_episodes
import torch
import pandas as pd
import sys
import concurrent.futures
sys.path.insert(0, './ae/imageAE')
from utilities import extract_embeddings

parser = argparse.ArgumentParser(description="Train the image autoencoder")

# Define command-line arguments
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--dir_path", type=str, help="Path to the dataset")
parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
parser.add_argument("--save_path", type=str, help="Path where to save")
parser.add_argument("--game", type=str, help="Game to consider")
parser.add_argument("--n_seed", type=int, help="Number of seed to consider")
parser.add_argument("--n_ckpt", type=int, help="Number of checkpoint to extract")
parser.add_argument("--normalize", type=int, help="Normalize data before computation")
parser.add_argument("--log_scale", type=int, help="Log scale data before computation")
parser.add_argument("--n_workers", type=int, help="Number of workers to use")

# Parse the command-line arguments
args = parser.parse_args()
device = args.device
dir_path = args.dir_path
model_path = args.model_path
save_path = args.save_path
game = args.game
n_seed = args.n_seed
n_ckpt = args.n_ckpt
normalize = args.normalize
log_scale = args.log_scale
n_workers = args.n_workers

final_df = None  # Initialize final_df outside the loop

seed = 42
seed_everything(seed, workers=True)

def process_checkpoint(seed, ckpt):
    return get_episodes(dir_path, seed, ckpt)

with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
    # List comprehension to submit tasks to the thread pool
    futures = [executor.submit(process_checkpoint, seed, ckpt) for seed in range(1, n_seed + 1) for ckpt in range(1, n_ckpt + 1)]
    
    # Iterate over completed futures and concatenate results
    for future in concurrent.futures.as_completed(futures):
        result_df = future.result()
        if final_df is None:
            final_df = result_df
        else:
            final_df = pd.concat([final_df, result_df], ignore_index=True)

filtered_df = final_df[final_df["Observation"].apply(lambda x: x.shape[0] == 4)].copy()

images_list = filtered_df["Observation"]

images_tensor = [torch.tensor(arr) for arr in images_list]
images_tensor = torch.stack(images_tensor)

embeddings = extract_embeddings(images_tensor, device, model_path, normalize, log_scale)

print(images_tensor.shape)
print(embeddings.shape)
print(len(embeddings.tolist()))
print(len(filtered_df["Observation"].to_list()))

filtered_df["Observation"] = embeddings.tolist()

#final_df.to_csv(f'{save_path}/{game}.csv', index=False, sep=';')
filtered_df.to_pickle(f'{save_path}/{game}.pkl')