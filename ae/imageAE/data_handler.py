import gzip
import numpy as np
import pandas as pd
import random

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