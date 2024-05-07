import argparse
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

def load_data(desc_flag, max_token_length=256, cut=512):
    
    games = ["Alien", "IceHockey", "Pong", "AirRaid", "SpaceInvaders"]
    data = None

    for game in games:
        file_path = f"../similarity_dataset/{game}.pkl"
        game_data = pd.read_pickle(file_path)
        if game == "Alien":
            text = "In Alien, players take on the role of a lone astronaut stranded aboard a space station infested with deadly aliens. \
                    The objective is to survive wave after wave of relentless alien attacks while attempting to escape the doomed station. \
                    The game features a top-down perspective, with the player navigating through maze-like corridors filled with lurking aliens. \
                    Armed with a limited supply of ammunition, players must strategically defend themselves against the alien onslaught. \
                    The tension mounts as the aliens grow increasingly aggressive and numerous with each passing level."
        elif game == "IceHockey":
            text = "In Ice Hockey, players take control of a team of ice hockey players in a thrilling one-on-one matchup against the computer or another player. \
                    The game features simple yet engaging gameplay mechanics, with players maneuvering their skaters across the icy rink to outmaneuver the opposition and score goals. \
                    Using the joystick controller, players can control the direction and speed of their skaters, as well as perform actions such as passing, shooting, and checking."
        elif game == "Pong":
            text = "In Pong, two players control paddles on either side of the screen, with the goal of hitting a ball back and forth between them. \
                    The paddles move vertically and are controlled by players using knobs or joysticks. \
                    The objective is to score points by successfully hitting the ball past the opponent's paddle. \
                    The game features simple graphics consisting of two-dimensional paddles and a square ball bouncing across the screen. \
                    As the game progresses, the ball's speed increases, making it more challenging for players to react and hit the ball."
        elif game == "AirRaid":
            text = "In Air Raid, players must fend off waves of enemy aircraft while strategically managing their ammunition and fuel supplies. \
                    The game unfolds with the player's jet stationed at the bottom of the screen, with a city skyline serving as the backdrop. \
                    Enemy aircraft, depicted as various geometric shapes, descend from the top of the screen in relentless waves, unleashing barrages of projectiles as they attempt to destroy the player's jet and the city below. \
                    Players control the fighter jet using the joystick controller, maneuvering left and right to evade enemy fire while returning fire to eliminate incoming threats. \
                    The game's challenging mechanics require players to balance offensive and defensive strategies, carefully conserving ammunition and fuel to survive each increasingly difficult wave of attacks."
        elif game == "SpaceInvaders":
            text = "In Space Invaders, players control a movable laser cannon situated at the bottom of the screen, aiming to eliminate descending formations of pixelated alien invaders. \
                    The invaders gradually advance towards the bottom of the screen, firing back at the player in a coordinated assault. \
                    As the player successfully shoots down aliens, the game's pace intensifies, with the remaining invaders moving faster and adopting more aggressive attack patterns. \
                    Players must strategically dodge incoming enemy fire while aiming carefully to eliminate as many aliens as possible."
        else:
            text = None

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        if desc_flag == 0:
            tokens = tokenizer.encode(text, max_length=max_token_length, truncation=True, padding='max_length')
            tokens = tokenizer.convert_tokens_to_ids(tokens)
            game_data['Text'] = [tokens] * len(game_data)

        elif desc_flag == 1:
            tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=256)
            model = BertModel.from_pretrained('bert-base-cased')
            with torch.no_grad():
                outputs = model(**tokens)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token embeddings
            embeddings = np.squeeze(embeddings)
            game_data['Text'] = [embeddings] * len(game_data)

        else:
            tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=256)
            model = BertModel.from_pretrained('bert-base-cased')
            # Forward pass to obtain BERT embeddings
            with torch.no_grad():
                outputs = model(**tokens)
                hidden_states = outputs.last_hidden_state

            # Calculate the mean of the hidden states across all tokens for each text
            mean_hidden_states = torch.mean(hidden_states, dim=1).numpy()
            mean_hidden_states = np.squeeze(mean_hidden_states)
            game_data['Text'] = [mean_hidden_states] * len(game_data)

            
        game_data['Game'] = [game] * len(game_data)
        if data is None:
            data = game_data
        else:
            data = pd.concat([data, game_data], ignore_index=True)

    # Exclude all the rows with an Action or a Reward smaller than cut
    data = data[data['Action'].apply(lambda x: len(x)) >= cut]
    data = data[data['Reward'].apply(lambda x: len(x)) >= cut]

    data['Action'] = data['Action'].apply(lambda x: x[len(x)-cut:])
    data['Reward'] = data['Reward'].apply(lambda x: x[len(x)-cut:])
    
    return data

def find_optimal_k(X, file_path, filename, max_k=30, n_jobs=-1):

    warnings.filterwarnings('ignore')

    def compute_kmeans_sse(k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        return kmeans.inertia_

    sse_list = Parallel(n_jobs=n_jobs)(delayed(compute_kmeans_sse)(k) for k in range(2, max_k + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k + 1), sse_list)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.savefig(fname=f'{file_path}/{filename}_game.png', format="png")

parser = argparse.ArgumentParser(description="Cluster data")

# Define command-line arguments
parser.add_argument("--description_type", type=int, help="Int value indicating the type of description to use (0 description tokenized, 1 description BERT CLS, 2 description BERT embeddings)")

args = parser.parse_args()
description_type = args.description_type

data = load_data(description_type)

# Function to concatenate elements of a list
def concatenate_elements(row):
    concatenated = []
    for column in data.columns:
        if column != 'Game':
            concatenated.extend(row[column])
    return concatenated

# Apply the function row-wise and create a new column
data['Feature'] = data.apply(concatenate_elements, axis=1)

# Convert lists in 'Feature' column to numpy arrays
data['Feature'] = data['Feature'].apply(lambda x: np.array(x))

X = np.vstack(data['Feature'])
find_optimal_k(X, ".", "best_k_concat_no_norm")

print("best_k_concat_no_norm done")

scaler = MinMaxScaler()
minmax_data = scaler.fit_transform(data['Feature'].to_list())

find_optimal_k(minmax_data, ".", "best_k_concat_min_max")

print("best_k_concat_min_max done")

feature_column = np.array(data['Feature'].tolist())
mean_features = np.mean(feature_column, axis=0)
std_features = np.std(feature_column, axis=0)
zero_indices = np.where(std_features == 0)[0]

if len(zero_indices) > 0:
    # Drop elements from the vector at zero indices
    mean_features = np.delete(mean_features, zero_indices)
    std_features = np.delete(std_features, zero_indices)
    # Drop columns with zero indices
    feature_column = np.delete(feature_column, zero_indices, axis=1)

zscore_data = np.array([(np.array(row) - mean_features) / std_features for row in feature_column])

find_optimal_k(zscore_data, ".", "best_k_concat_zscore")

print("best_k_concat_zscore done")

feature_matrix = np.array(data['Feature'].tolist())
covariance_matrix = np.cov(feature_matrix, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
# Select number of components based on eigenvalues using the number of components that explain n% of the variance
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_variance_ratio >= 0.80) + 1

# Perform PCA on the first 51 principal components
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(feature_matrix)

find_optimal_k(principal_components, ".", "best_k_concat_80")

print("best_k_concat_80 done")

num_components = np.argmax(cumulative_variance_ratio >= 0.90) + 1

# Perform PCA on the first 214 principal components
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(feature_matrix)

find_optimal_k(principal_components, ".", "best_k_concat_90")

print("best_k_concat_90 done")

