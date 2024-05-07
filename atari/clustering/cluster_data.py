import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from transformers import BertTokenizer, BertModel
import os
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import torch
import plotly.express as px
import plotly.io as pio

# Function to concatenate elements of a list
def concatenate_elements(data, row):
    concatenated = []
    for column in data.columns:
        if column != 'Game':
            concatenated.extend(row[column])
    return concatenated

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

def apply_kmeans(data, column, n_clusters, file_path, filename, data_type):

    X = np.vstack(data[column])

    # Fit KMeans clustering
    num_clusters = n_clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print("Silhouette Score:", silhouette_avg)

    if data_type < 2:
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=3)
        feature_pca = pca.fit_transform(X)

        # Visualize clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=feature_pca[:, 0], y=feature_pca[:, 1], hue=data['Game'], palette='viridis', legend='full')

        plt.title(filename)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Game')
        plt.savefig(fname=f'{file_path}/{filename}_game.png', format = "png")

        fig = px.scatter_3d(
            feature_pca, x=0, y=1, z=2, color=data['Game'],
            title=filename,
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )

        # Save Plotly figure as HTML
        pio.write_html(fig, f'{file_path}/{filename}_game.html')

        # Visualize clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=feature_pca[:, 0], y=feature_pca[:, 1], hue=kmeans.labels_, palette='viridis', legend='full')

        plt.title(filename)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Game')
        plt.savefig(fname=f'{file_path}/{filename}_clusters.png', format = "png")

        fig = px.scatter_3d(
            feature_pca, x=0, y=1, z=2, color=kmeans.labels_,
            title=filename,
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )

        # Save Plotly figure as HTML
        pio.write_html(fig, f'{file_path}/{filename}_clusters.html')

    else:

        # Visualize clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=data['Game'], palette='viridis', legend='full')

        plt.title(filename)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Game')
        plt.savefig(fname=f'{file_path}/{filename}_game.png', format = "png")

        fig = px.scatter_3d(
            X, x=0, y=1, z=2, color=data['Game'],
            title=filename,
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )

        # Save Plotly figure as HTML
        pio.write_html(fig, f'{file_path}/{filename}_game.html')

        # Visualize clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans.labels_, palette='viridis', legend='full')

        plt.title(filename)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Game')
        plt.savefig(fname=f'{file_path}/{filename}_clusters.png', format = "png")

        fig = px.scatter_3d(
            X, x=0, y=1, z=2, color=kmeans.labels_,
            title=filename,
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )

        # Save Plotly figure as HTML
        pio.write_html(fig, f'{file_path}/{filename}_clusters.html')

    return kmeans

def apply_tsne(data, column, file_path, filename, kmeans):

    # Convert 'Feature' column into a 2D numpy array
    X = np.vstack(data[column])
    tsne = TSNE(n_components=2, random_state=42)
    # Fit and transform the data
    tsne_result = tsne.fit_transform(X)

    plt.figure(figsize=(10, 6))

    # Define colors based on 'game' column
    colors = {
        'Pong': 'blue',
        'Alien': 'red',
        'AirRaid': 'green',
        'IceHockey': 'yellow',
        'SpaceInvaders' : 'cyan'
    }

    # Iterate over unique games and plot each with a different color
    for game, color in colors.items():
        indices = data['Game'] == game
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], color=color, label=game, alpha=0.5)

    plt.title(filename)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Game')
    plt.savefig(fname= f'{file_path}/{filename}_game.png', format = "png")

    plt.figure(figsize=(10, 6))

    # Get unique cluster labels
    unique_labels = sorted(set(kmeans.labels_))

    cmap = plt.cm.get_cmap('viridis', len(unique_labels))

    # Plot data points with different colors based on KMeans labels
    for label, color in zip(unique_labels, colors):
        indices = kmeans.labels_ == label
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], color=cmap(label), label=f'Cluster {label}', alpha=0.5)


    plt.title('t-SNE Visualization of images embedding')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Cluster')
    plt.savefig(fname= f'{file_path}/{filename}_cluster.png', format = "png")

def compute_cosine_similarity(data, column, file_path, filename):

    games = data['Game'].unique()

    game_similarity = pd.DataFrame(index=games, columns=games)

    for i in range(len(games)):
        for j in range(len(games)):

            combined_vectors_game1 = np.stack(data[data['Game'] == games[i]][column].tolist())
            combined_vectors_game2 = np.stack(data[data['Game'] == games[j]][column].tolist())

            # Compute cosine similarity matrix
            cosine_sim_matrix = cosine_similarity(combined_vectors_game1, combined_vectors_game2)
            
            # Take the mean cosine similarity
            mean_cosine_similarity = np.mean(cosine_sim_matrix)

            game_similarity.loc[games[i], games[j]] = mean_cosine_similarity

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(game_similarity.astype(float), annot=True, cmap='coolwarm', linewidths=.5)
    plt.title(filename)
    plt.xlabel('Game')
    plt.ylabel('Game')
    plt.savefig(fname= f'{file_path}/{filename}.png', format = "png")

#Data type 0 == only emb
#Data type 1 == emb. concat to dataset
#Data type 2 == k principal components
#Normalization type 0 == no norm
#Normalization type 1 == min-max scaling
#Normalization type 2 == z-score scaling
def exec_combo(data, column, directory, data_type, normalization_type, n_clusters, k_pc = None):

    filename = ""

    if data_type == 2:
        data_str = "k_pc"
        filename = data_str

        filepath = f'{directory}/{data_str}'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            print(f"Directory '{filepath}' created successfully")

        if k_pc is not None:
            filepath = f'{directory}/{data_str}/{k_pc}'
            filename = f'{filename}_{k_pc}'
            if not os.path.exists(filepath):
                os.makedirs(filepath)
                print(f"Directory '{filepath}' created successfully")
    else:
        if data_type == 0:
            data_str = "only_emb"
        elif data_type == 1:
            data_str = "emb+dataset"

        filename = data_str

        filepath = f'{directory}/{data_str}'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            print(f"Directory '{filepath}' created successfully")

    if normalization_type == 0:
        norm_str = "no_norm"
    elif normalization_type == 1:
        norm_str = "minmax"
    else:
        norm_str = "z-score"

    filename = f'{filename}_{norm_str}'
    filepath = f'{filepath}/{norm_str}'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print(f"Directory '{filepath}' created successfully")

    #Apply kmeans
    if not os.path.exists(f'{filepath}/kmeans'):
        os.makedirs(f'{filepath}/kmeans')
        print(f"Directory '{filepath}/kmeans' created successfully")

    kmeans = apply_kmeans(data, column, n_clusters, f'{filepath}/kmeans', f'kmeans_{filename}', data_type)

    #Apply TSNE
    if not os.path.exists(f'{filepath}/tsne'):
        os.makedirs(f'{filepath}/tsne')
        print(f"Directory '{filepath}/tsne' created successfully")

    apply_tsne(data, column, f'{filepath}/tsne', f'tsne_{filename}', kmeans)

    #Apply cosine similarity
    if not os.path.exists(f'{filepath}/cosine'):
        os.makedirs(f'{filepath}/cosine')
        print(f"Directory '{filepath}/cosine' created successfully")

    compute_cosine_similarity(data, column, f'{filepath}/cosine', f'cosine_similarity_{filename}')

def apply_minmax(data, column, new_col):
    scaler = MinMaxScaler()
    data[new_col] = scaler.fit_transform(data[column].to_list()).tolist()

def apply_zscore(data, column, new_col):
    feature_column = np.array(data[column].tolist())
    mean_features = np.mean(feature_column, axis=0)
    std_features = np.std(feature_column, axis=0)
    zero_indices = np.where(std_features == 0)[0]
    if len(zero_indices) > 0:
        # Drop elements from the vector at zero indices
        mean_features = np.delete(mean_features, zero_indices)
        std_features = np.delete(std_features, zero_indices)
        # Drop columns with zero indices
        feature_column = np.delete(feature_column, zero_indices, axis=1)

    data[new_col] = [(np.array(row) - mean_features) / std_features for row in feature_column]

def extract_k_principal_components(data, column, new_col, k):

    feature_matrix = np.array(data[column].tolist())
    pca = PCA(n_components=k)
    principal_components = pca.fit_transform(feature_matrix)

    data[new_col] = principal_components.tolist()


parser = argparse.ArgumentParser(description="Cluster data")

# Define command-line arguments
parser.add_argument("--description_type", type=int, help="Int value indicating the type of description to use (0 description tokenized, 1 description BERT CLS, 2 description BERT embeddings)")

args = parser.parse_args()
description_type = args.description_type

data = load_data(description_type)

directory = "./clustering_res"

if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully")

if description_type == 0:
    directory = f'{directory}/tokens'
elif description_type == 1:
    directory = f'{directory}/CLS'
else:
    directory = f'{directory}/BERT_embeddings'

if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully")


# Only embeddings, no normalization
exec_combo(data, 'Observation', directory, 0, 0, n_clusters=5)
print("only emb done")

# Embeddings concat to dataset, no normalization
data['Features'] = data.apply(lambda r: concatenate_elements(data, r), axis=1)
data['Features'] = data['Features'].apply(lambda x: np.array(x))
exec_combo(data, 'Features', directory, 1, 0, n_clusters=5)
print("concat no norm done")

# Embeddings concat to dataset, min-max scaling
apply_minmax(data, 'Features', 'Features_minmax')
exec_combo(data, 'Features_minmax', directory, 1, 1, n_clusters=5)
print("concat minmax done")

# Embeddings concat to dataset, z-score scaling
apply_zscore(data, 'Features', 'Features_zscore')
exec_combo(data, 'Features_zscore', directory, 1, 2, n_clusters=5)
print("concat zscore done")

# 90 principal components, no normalization
extract_k_principal_components(data, 'Features', '90_pc', 214)
exec_combo(data, '90_pc', directory, 2, 0, n_clusters=5, k_pc=90)
print("90pc no norm done")

# 90 principal components, min-max scaling
apply_minmax(data, '90_pc', '90_pc_minmax')
exec_combo(data, '90_pc_minmax', directory, 2, 1, n_clusters=5, k_pc=90)
print("90pc minmax done")

# 90 principal components, z-score scaling
apply_zscore(data, '90_pc', '90_pc_zscore')
exec_combo(data, '90_pc_zscore', directory, 2, 2, n_clusters=5, k_pc=90)
print("90pc zscore done")

# 80 principal components, no normalization
extract_k_principal_components(data, 'Features', '80_pc', 51)
exec_combo(data, '80_pc', directory, 2, 0, n_clusters=5, k_pc=80)
print("80pc no norm done")

# 80 principal components, min-max scaling
apply_minmax(data, '80_pc', '80_pc_minmax')
exec_combo(data, '80_pc_minmax', directory, 2, 1, n_clusters=5, k_pc=80)
print("80pc minmax done")

# 80 principal components, z-score scaling
apply_zscore(data, '80_pc', '80_pc_zscore')
exec_combo(data, '80_pc_zscore', directory, 2, 2, n_clusters=5, k_pc=80)
print("80pc zscore done")