from tqdm import tqdm
import numpy as np
from copy import deepcopy
import os
import torch
import gymnasium as gym
import pickle
import argparse
import sys
sys.path.insert(1, "..")
from stable_baselines3.sac.sac import SACMaster


def eval_moodel(test_env, agent, n_episodes):
    next_lane_dict={
        ('a', 'b'): ('b', 'c'),
        ('b', 'c'): ('c', 'd'),
        ('c', 'd'): ('d', 'e'),
        ('d', 'e'): ('e', 'f'),
        ('e', 'f'): ('f', 'g'),
        ('f', 'g'): ('g', 'h'),
        ('g', 'h'): ('h', 'i'),
        ('h', 'i'): ('i', 'a'),
        ('i', 'a'): ('a', 'b')
    }
    all_passed_sections = []
    all_mean_dist_per_action = []
    #How many actions were performed in a single section
    per_sector_actions = []

    section_passed = [0]*10

    section_passed_start_from = {
        ('a', 'b'): 0,
        ('b', 'c'): 0,
        ('c', 'd'): 0,
        ('d', 'e'): 0,
        ('e', 'f'): 0,
        ('f', 'g'): 0,
        ('g', 'h'): 0,
        ('h', 'i'): 0,
        ('i', 'a'): 0
    }

    section_activated_skill = {
        ('a', 'b'): np.zeros(4),
        ('b', 'c'): np.zeros(4),
        ('c', 'd'): np.zeros(4),
        ('d', 'e'): np.zeros(4),
        ('e', 'f'): np.zeros(4),
        ('f', 'g'): np.zeros(4),
        ('g', 'h'): np.zeros(4),
        ('h', 'i'): np.zeros(4),
        ('i', 'a'): np.zeros(4),
    }
    section_staying_counter = {
        ('a', 'b'): 0,
        ('b', 'c'): 0,
        ('c', 'd'): 0,
        ('d', 'e'): 0,
        ('e', 'f'): 0,
        ('f', 'g'): 0,
        ('g', 'h'): 0,
        ('h', 'i'): 0,
        ('i', 'a'): 0,
    }

    dist_list = []
    for video in tqdm(range(n_episodes)):
        done = truncated = False
        obs, info = test_env.reset(seed=video)
        prev_pos = deepcopy(info["position"])
        road_net = info["road"]
        distance = 0
        dist = 0
        episode_length = 0
        passed_sections = 0
        n_actions = 0
        lane_edge_dict = {}
        for key, value in road_net.lanes_dict().items():
            lane_edge_dict[value] = (key[0], key[1])
            
        current_lane = lane_edge_dict[info['current_lane']]
        next_lane = next_lane_dict[current_lane]
        
        spawining_lane = current_lane
        
        while not (done or truncated):
            action, _states = agent.predict(obs)
            section_staying_counter[current_lane] += 1
            obs, reward, done, truncated, info = test_env.step(action)
            
            # Compute distance
            curr_pos = info['position']
            if prev_pos is not None:
                if info['speed'] >= 0:
                    dist += np.linalg.norm(curr_pos - prev_pos)
                else:
                    dist -= np.linalg.norm(curr_pos - prev_pos)


            distance += np.linalg.norm(prev_pos - curr_pos)
            episode_length += 1
            n_actions += 1
            current_lane = lane_edge_dict[info['current_lane']]
            if current_lane == next_lane:
                next_lane = next_lane_dict[current_lane]
                passed_sections += 1
                per_sector_actions.append(n_actions)
                n_actions = 0
            
            prev_pos = copy(curr_pos)

        dist_list.append(dist)
        all_passed_sections.append(passed_sections)  
        all_mean_dist_per_action.append(distance/episode_length)          
        section_passed_start_from[spawining_lane] += passed_sections
        if passed_sections < 10:
            section_passed[passed_sections] += 1
        else:
            section_passed[-1] += 1

    for k, v in section_activated_skill.items():
        v /= section_staying_counter[k]

    return all_mean_dist_per_action, section_staying_counter, section_activated_skill, section_passed_start_from, per_sector_actions, all_passed_sections, dist_list


parser = argparse.ArgumentParser(description="Evaluate RL transfer")

# Define command-line arguments
parser.add_argument("--env_name", type=str, help="Environment to use")
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--models_path", type=str, help="Path to the trained models")
parser.add_argument("--save_path", type=str, help="Path where to save the metrics")
parser.add_argument("--n_episodes", type=int, help="Number of episodes to perform")

# Parse the command-line arguments
args = parser.parse_args()

env_name = args.env_name
device = args.device
models_path = args.models_path
save_path = args.save_path
n_episodes = args.n_episodes

KINEMATICS_OBSERVATION = {
    "type": "Kinematics",
    "vehicles_count": 5,
    "features": ["presence", "x", "y", "vx", "vy", "heading", "long_off", "lat_off", "ang_off"],
    "absolute": False,
    "order": "sorted",
}

config = {
    "observation": KINEMATICS_OBSERVATION,
    "action": {
        "type": "ContinuousAction",
    },
    "policy_frequency": 5, 
    "vehicles_count": 5,
    # "real_time_rendering":True,
}


if device != "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    device = torch.device(0)

env = gym.make("racetrack-v0", config = config)

dir = os.listdir(models_path)

res_list = []

for file in dir:
    tokens = file.split("_")

    agent = SACMaster.load(f"{models_path}/{file}")
    print(file)
    all_mean_dist_per_action, section_staying_counter, section_activated_skill, section_passed_start_from, per_sector_actions, all_passed_sections, dist_list = eval_moodel(env, agent, n_episodes)
    res_dict = {
        'model': file,
        'all_mean_dist_per_action': all_mean_dist_per_action,
        'section_staying_counter': section_staying_counter,
        'section_activated_skill': section_activated_skill,
        'section_passed_start_from': section_passed_start_from,
        'per_section_actions': per_sector_actions,
        'all_passed_sections': all_passed_sections,
        "dist_list": dist_list
    }

    res_list.append(res_dict)

# Check if the file exists
if os.path.exists(save_path):
    # Load existing data
    with open(save_path, 'rb') as file:
        existing_data = pickle.load(file)
else:
    # If the file does not exist, initialize an empty list
    existing_data = []

# Append the new data
existing_data.extend(res_list)

# Save the updated data back to the pickle file
with open(save_path, 'wb') as file:
    pickle.dump(existing_data, file)