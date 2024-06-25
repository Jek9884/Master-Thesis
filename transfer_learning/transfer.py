import torch
import os
import gymnasium as gym
import sys
sys.path.insert(1, "../")
from stable_baselines3.sac.sac import SAC, SACMaster
from stable_baselines3.common.env_util import make_vec_env
from ae.imageAE.highway_model import HighwayEnvModel
import time
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

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

env_name = "racetrack-v0"

# Without transfer learning
env = make_vec_env(env_name, n_envs=32)

# With transfer learning
#env = gym.make(env_name, render_mode='rgb_array')

seed = 42
device = "2"
policy_dir = "../highway-env/subpolicies"
experience_dir = "./transfer_dataset"
description_dir = "./descriptions.json"  
ae_path = "../highway-env/final_model/best_highway_batch128_wd1e-7_lr1e-5_holdout.ckpt"
#task_description = "boh"

if device != "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    device = torch.device(0)

ae = HighwayEnvModel.load_from_checkpoint(ae_path)

total_timesteps = 1e7
progress_bar_callback = ProgressBarCallback(total_timesteps)

# Without transfer learning
agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=1e-4, batch_size=1024, tau=0.9, gamma=0.99, gradient_steps=10, 
            use_sde=False, seed=seed)

"""# With task description
agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=1e-4, batch_size=1024, tau=0.9, gamma=0.99, gradient_steps=10, 
                    use_sde=False, policy_dir=policy_dir, experience_dir=experience_dir, 
                    descriptions_dir=description_dir, ae_model=ae, k=20, similarity_thr=0.6, task_description=task_description,
                    tokenizer_str="bert-base-cased")"""


"""# Without task description
agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=1e-4, batch_size=1024, tau=0.9, gamma=0.99, gradient_steps=10, 
                    use_sde=False, policy_dir=policy_dir, experience_dir=experience_dir, descriptions_dir=description_dir, 
                    ae_model=ae, k=20, similarity_thr=0.6, device=device)"""

start_time = time.time()
agent.learn(total_timesteps=total_timesteps, callback=progress_bar_callback)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

agent.save("racetrack_no_transfer_16env")