import pickle
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import multiprocessing

def find_pkls_in_subfolders(directory):
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def apply_upsampling(image):
        
    # Dimensions of the original image
    original_height, original_width = image.shape

    # New dimensions
    new_height, new_width = 600, 600
    diff_height = new_height - original_height
    diff_width = new_width - original_width

    # Calculate the padding sizes for height and width
    top_pad = (new_height - original_height) // 2
    bottom_pad = new_height - original_height - top_pad
    left_pad = (new_width - original_width) // 2
    right_pad = new_width - original_width - left_pad
    
    background_color = 100

    # Create a new larger matrix filled with the padding value
    larger_matrix = np.full((new_height, new_width), background_color, dtype=image.dtype)
    
    # Copy the original image into the center of the larger matrix
    larger_matrix[top_pad:top_pad + original_height, left_pad:left_pad + original_width] = image

    return larger_matrix

def process_episode(images):

    upsampled_images = []

    for i in range(images.shape[0]):
        upsampled_images.append(apply_upsampling(images[i]))

    return np.array(upsampled_images)


parser = argparse.ArgumentParser(description="Extract images from the full dataset to train the Autoencoder", allow_abbrev=False)

# Define command-line arguments
parser.add_argument("--dataset_path", type=str, help="Path of the dataset")
parser.add_argument("--save_path", type=str, help="Path where to save the extracted data")
parser.add_argument("--n_sample", type=float, default=1, help="Percentage of sample to consider for each task")

args = parser.parse_args()
dataset_path = args.dataset_path
save_path = args.save_path
n_sample = args.n_sample

if n_sample <= 0 or n_sample > 1:
    print("Percentage out of range (0,1]")
else:

    if not os.path.exists(dataset_path):
        print(f"File does not exist")
    else:
        tensor_list = None

        pkl_files = find_pkls_in_subfolders(dataset_path)

        for file in pkl_files:

            split_string = file.split('/')
            env_name = split_string[2]

            # Exclude u-turn for observations dimension issues
            if env_name != "u-turn-v0":

                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    obs = data['Observation']
                    obs = np.array(obs)
                    if n_sample < 1:
                        n_obs = int(obs.shape[0] * n_sample)
                        print(f'Env name: {env_name} - Num sample {n_obs}')
                        obs = obs[:n_obs, :, :, :]
                    result = obs

                    # Make upsampling only for observations that are smaller than 600x600
                    if obs[0][0].shape[0] != 600 or obs[0][0].shape[1] != 600:

                        num_processes = multiprocessing.cpu_count()

                        # Create a pool of workers
                        pool = multiprocessing.Pool(processes=num_processes)

                        # Map the function to the collection of items using multiple processes
                        result = pool.map(process_episode, obs)

                        # Close the pool to release resources
                        pool.close()
                        pool.join()

                        result = np.array(result)
                        
                if tensor_list is None:
                    tensor_list = result
                else:
                    tensor_list = np.concatenate((tensor_list, result))

        tensor = torch.tensor(tensor_list)

        # Save tensor to a file
        torch.save(tensor, f'{save_path}/final_res_{n_sample}.pt')




                    

            

            
            

            
