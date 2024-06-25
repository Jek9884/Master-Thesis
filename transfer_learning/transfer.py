import torch
import os
import sys
sys.path.insert(1, "../")
from stable_baselines3.sac.sac import SACMaster
from stable_baselines3.common.env_util import make_vec_env
from ae.imageAE.highway_model import HighwayEnvModel
import time
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import argparse

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

parser = argparse.ArgumentParser(description="Train a RL agent using TL")

# Define command-line arguments
parser.add_argument("--env_name", type=str, help="Environment to use")
parser.add_argument("--policy_dir", type=str, help="Path to the source policies")
parser.add_argument("--experience_dir", type=str, help="Path to the source dataset")
parser.add_argument("--description_dir", type=str, help="Path to the source tasks descriptions")
parser.add_argument("--ae_path", type=str, help="Path to the ae model")
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--transfer_type", type=int, help="0 for no transfer CV or 1 for transfer")
parser.add_argument("--total_timesteps", type=int, help="Max number of timesteps to perform")
parser.add_argument("--n_envs", type=int, default=1, help="Number of environments to use in parallel")
parser.add_argument("--seed", type=int, default=42, help="Seed to use for replicability")
parser.add_argument("--k", type=int, help="Number of time-steps to perform before to use the metric")
parser.add_argument("--similarity_thr", type=float, help="Threshold for the metric")

# Parse the command-line arguments
args = parser.parse_args()

assert args.transfer_type == 0 or args.transfer_type == 1
assert args.n_envs > 0
assert args.similarity_thr >= -1 and args.similarity_thr <= 1
assert args.k > 0

env_name = args.env_name#"racetrack-v0"
policy_dir = args.policy_dir
experience_dir = args.experience_dir
description_dir = args.description_dir
ae_path = args.ae_path
device = args.device
transfer_type = args.transfer_type
total_timesteps = args.total_timesteps
n_envs = args.n_envs
seed = args.seed
k = args.k
similarity_thr = args.similarity_thr

task_description = None

# Init env
env = make_vec_env(env_name, n_envs=n_envs)

if device != "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    device = torch.device(0)

ae = HighwayEnvModel.load_from_checkpoint(ae_path)

progress_bar_callback = ProgressBarCallback(total_timesteps)

if transfer_type == 0:
    # Without transfer learning
    agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=1e-4, batch_size=1024, tau=0.9, gamma=0.99, gradient_steps=10, 
                        use_sde=False, device=device, seed=seed)
else:
    if task_description is not None:
        # TL with task description
        agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=1e-4, batch_size=1024, tau=0.9, gamma=0.99, gradient_steps=10, 
                            use_sde=False, policy_dir=policy_dir, experience_dir=experience_dir, 
                            descriptions_dir=description_dir, ae_model=ae, k=k, similarity_thr=similarity_thr, task_description=task_description,
                            tokenizer_str="bert-base-cased", device=device, seed=seed)
    else:
        # TL without task description
        agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=1e-4, batch_size=1024, tau=0.9, gamma=0.99, gradient_steps=10, 
                            use_sde=False, policy_dir=policy_dir, experience_dir=experience_dir, 
                            descriptions_dir=description_dir, ae_model=ae, k=k, similarity_thr=similarity_thr, device=device, seed=seed)

agent.learn(total_timesteps=total_timesteps, callback=progress_bar_callback)

if transfer_type == 1:
    agent.save(f"{env_name}_{transfer_type}_{n_envs}env_{k}_{similarity_thr}")
else:
    agent.save(f"{env_name}_{transfer_type}_{n_envs}env")
