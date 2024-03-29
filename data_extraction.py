import pandas as pd
import numpy as np
import gzip
import os
import argparse

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

def get_episodes(dir_path, seed, ckpt):

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

            data = {
                'Observation' : [obs[0:idx_lst[0]+1, :, :][-4:, :, :]],
                'Action' : [act[0:idx_lst[0]+1].tolist()],
                'Reward' : [rwd[0:idx_lst[0]+1].tolist()]
            }

            for i in range(1,len(idx_lst)):
                start = idx_lst[i-1] + 1
                end = idx_lst[i] + 1
                data['Observation'].append(obs[start:end, :, :][-4:, :, :])
                data['Action'].append(act[start:end].tolist())
                data['Reward'].append(rwd[start:end].tolist())

            return pd.DataFrame(data)
        
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
    
def normalize_images_tensor(input_tensor):
    return input_tensor / 255.0