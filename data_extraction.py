import pandas as pd
import numpy as np
import gzip
import os
import argparse

def get_episodes_idx(dir_path, seed, ckpt):

    filename_term = f"{dir_path}/{seed}/replay_logs/$store$_terminal_ckpt.{ckpt}.gz"
    pickled_data = gzip.GzipFile(filename=filename_term)
    np_term = np.load(pickled_data)

    episodes_idx = []

    for idx, elem in enumerate(np_term):
        if elem == 1:
            episodes_idx.append(idx)

    return episodes_idx

def get_best_episodes(dir_path, seed, ckpt, k):

    idx_lst = get_episodes_idx(dir_path, seed, ckpt)
    
    filename_obs = f"{dir_path}/{seed}/replay_logs/$store$_observation_ckpt.{ckpt}.gz"
    pickled_data = gzip.GzipFile(filename=filename_obs)
    obs = np.load(pickled_data)
    filename_act = f"{dir_path}/{seed}/replay_logs/$store$_action_ckpt.{ckpt}.gz"
    pickled_data = gzip.GzipFile(filename=filename_act)
    act = np.load(pickled_data)
    filename_rwd = f"{dir_path}/{seed}/replay_logs/$store$_reward_ckpt.{ckpt}.gz"
    pickled_data = gzip.GzipFile(filename=filename_rwd)
    rwd = np.load(pickled_data)

    data = {
        'Observation' : [np.array([obs[0:idx_lst[0]+1, :, :]])],
        'Action' : [np.array([act[0:idx_lst[0]+1]])],
        'Reward' : [np.array([rwd[0:idx_lst[0]+1]])]
    }

    for i in range(1,len(idx_lst)):
        start = idx_lst[i-1] + 1
        end = idx_lst[i] + 1
        data['Observation'].append(obs[start:end, :, :])
        data['Action'].append(act[start:end])
        data['Reward'].append(rwd[start:end])


    df = pd.DataFrame(data)
    df['Rank'] = df.apply(lambda row: row['Reward'].sum(), axis=1)
    df = df.sort_values(by='Rank', ascending=False)
    df = df.head(k)

    return df

def extract_seed_data(dir_path, save_path, seed, n_ckpt, k):

    all_data = pd.DataFrame({'Observation' : [], 'Action' : [], 'Reward' : []})

    # Create a file for each checkpoint with the selected episodes
    for ckpt in range(n_ckpt):

        res = get_best_episodes(dir_path, seed, ckpt, k)

        res.to_pickle(f'{save_path}/Ckpt{ckpt}.pkl')

        print(f"Checkpoint {ckpt} done")

    # Create single file with the best episodes
    #TODO remove Rank column?
    for ckpt in range(n_ckpt):
    
        filename = f"{save_path}/Ckpt{ckpt}.pkl"
        df = pd.read_pickle(filename)
        all_data = pd.concat([all_data, df], ignore_index=True)

        try:
            os.remove(filename)
            print(f"File {filename} removed successfully.")
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except PermissionError:
            print(f"Permission error: Unable to remove {filename}.")
        except Exception as e:
            print(f"An error occurred: {e}")

    all_data.drop()
    all_data.to_pickle(f"{save_path}/Seed{seed}.pkl")

def extract_all_games_data(dataset_path, save_path, seed, k):

    all_games = os.listdir(dataset_path)

    # Filter out only the directories
    folders = [item for item in all_games if os.path.isdir(os.path.join(dataset_path, item))]

    n_ckpt = 51

    for folder in folders:

        dir_path = f"{dataset_path}/{folder}"
        extract_seed_data(dir_path, save_path, seed, n_ckpt, k)
        print(f"{folder} done")


parser = argparse.ArgumentParser(description='Description of your script.')

# Add arguments
parser.add_argument('--dataset_path', type=str, help='Path to the dataset directory')
parser.add_argument('--save_path', type=str, help='Path to the directory where to save the files')
parser.add_argument('--seed', type=int, default=1, help='Agent seed to consider')
parser.add_argument('--k', type=int, default=10, help='Number of episodes to save for each checkpoint')

# Parse the command line arguments
args = parser.parse_args()

extract_all_games_data(args.dataset_path, args.save_path, args.seed, args.k)