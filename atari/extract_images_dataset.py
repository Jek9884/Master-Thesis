import torch
from concurrent.futures import ThreadPoolExecutor
from data_extraction import get_episodes_idx
import argparse
import gzip
import numpy as np
import pandas as pd
import os

def get_obs_episodes(dir_path, seed, ckpt):

    idx_lst = get_episodes_idx(dir_path, seed, ckpt)

    try:
        if len(idx_lst) > 0:
            filename_obs = f"{dir_path}/{seed}/replay_logs/$store$_observation_ckpt.{ckpt}.gz"
            pickled_data = gzip.GzipFile(filename=filename_obs)
            obs = np.load(pickled_data)

            data = {
                'Observation' : np.array([np.array(obs[0:idx_lst[0]+1, :, :][-4:, :, :])])
            }

            for i in range(1,len(idx_lst)):
                start = idx_lst[i-1] - 3
                end = idx_lst[i] + 1
                #print(obs[start:end, :, :][-4:, :, :].shape)
                episode = np.array([obs[start:end, :, :][-4:, :, :]])
                data['Observation'] = np.concatenate((data['Observation'], episode), axis=0)
            
            data['Observation'] = data['Observation'].tolist()
            df = pd.DataFrame(data)

            return df

        else:
            return pd.DataFrame({"Observation": [], "Action": [], "Reward": []})
                
    except FileNotFoundError:
        print(f"Checkpoint {ckpt} not found.")
        return pd.DataFrame({"Observation": [], "Action": [], "Reward": []})
    except PermissionError:
        print(f"Permission error: Unable to read {ckpt}.")
        return pd.DataFrame({"Observation": [], "Action": [], "Reward": []})
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame({"Observation": [], "Action": [], "Reward": []})


parser = argparse.ArgumentParser(description="Extract images dataset from the experience replay dataset")

parser.add_argument("--dir", type=str, help="Path to the dataset")
parser.add_argument("--game", type=str, help="What game to extract")
parser.add_argument("--n_seed", type=int, help="Number of seeds to use")
parser.add_argument("--n_ckpt", type=int, help="Number of checkpoints to use")

args = parser.parse_args()

directory = args.dir
game = args.game
n_seed = args.n_seed
n_ckpt = args.n_ckpt

dir_path = f"{directory}/{game}"

images_tensor = torch.empty(0, dtype=torch.float32)


def process_checkpoint(game_path, seed, ckpt):
    df = get_obs_episodes(game_path, seed, ckpt)
    images_list = df['Observation'].tolist()
    images_tensor = torch.tensor(images_list)
    return images_tensor

final_images_tensor = None

for seed in range(1,n_seed+1):

    with ThreadPoolExecutor() as executor:
        params = [(f"{directory}/{game}", seed, ckpt) for ckpt in range(n_ckpt)]
        
        # Submit tasks to executor
        results = [executor.submit(process_checkpoint, *param) for param in params]

    if final_images_tensor is None:
        final_images_tensor = torch.cat([future.result() for future in results], dim=0)
    else:
        data = torch.cat([future.result() for future in results], dim=0)
        final_images_tensor = torch.cat((final_images_tensor, data), dim=0)

print(final_images_tensor.shape)

file_path = f"./dataset_{game}_{n_ckpt}_{n_seed}.pt"
torch.save(final_images_tensor, file_path)