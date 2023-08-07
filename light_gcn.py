#!/usr/bin/env python
# coding: utf-8

# # **(X)CS224W - End to End - Preprocessing**

# 

# In[ ]:


import gzip
import os
import os.path as osp
from operator import itemgetter
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch_geometric.nn.models.lightgcn import LightGCN
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)
from tqdm import tqdm, trange

from e2e.datasets import SpotifyMPDataset

# In[ ]:

# Load the Spotify MDP from the file system
spotify_dataset = SpotifyMPDataset(
    root='./spotify_mpd', 
    url=f"file://{osp.join(Path('.').resolve(), 'spotify_preprocessed_dataset')}"
)


# In[ ]:

# Setup the device
device = torch.device("cpu")
if torch.cuda.is_available(): 
    device = torch.device("cuda")

print(f'Runing on device: {device}')

# In[ ]:

train_data = spotify_dataset[0]
test_data = spotify_dataset[1]

args = {
    'device': device,
    'num_layers': 2,
    'hidden_dim': train_data.num_node_features,
    'train_chunk_size': 2,
    'test_chunk_size': 4,
    'num_workers': 12,
    'dropout': 0.5,
    'lr': 0.001,
    'epochs': 10,
    'lambda_reg': 1e-4,
    'k': 600
}

print(args)

# In[ ]:

# Use the [LightGCN](https://arxiv.org/pdf/2002.02126.pdf) recommender model
model = LightGCN(
    num_nodes=test_data.num_nodes,
    embedding_dim=train_data.num_node_features,
    num_layers=args["num_layers"]
)

# In[ ]:

# Initialize the embeddings with the initial features from the full graph
model.embedding.weight.data = test_data["x"]


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])


# In[ ]:

r"""Generate the output file in the format required by the Spotify MPD Challenge.

Args:
    recommended_track_ids : Recommended tracks for the test nodes.
    edge_index: Adjacency matrix.
    filename: Name of the file to be generated.

Returns: 
    

"""
def generate_output_file(recommended_track_ids, edge_index, filename):

    lines = ['\n']

    team_line = "team_info, SH_XCS224W_Summer_2023, hurubaru@stanford.edu"

    lines.append(team_line)
    
    for i, playlist_node_id in enumerate(test_data["test_nodes_index"].numpy()):

        # Transform the recommended tracks for the current playlist to a list
        playlist_track_recommendations = list((recommended_track_ids[i]).numpy())

        # Exclude the pre-given tracks from the test playlist as per the challenge
        excluded_track_nodes_mask = edge_index[0] == playlist_node_id

        playlist_excluded_tracks = edge_index[1, torch.nonzero(excluded_track_nodes_mask).squeeze()]

        valid_recommended_tracks = list(filter(lambda t: t not in playlist_excluded_tracks, playlist_track_recommendations))

        # From the rest of the tracks recommended extract the top 500
        top_500_recommended_track_ids = valid_recommended_tracks[:500]

        pid = str(spotify_dataset.playlist_id_map.get(str(playlist_node_id)))

        # Get the tracks URI given the tracks id
        top_500_recommended_track_uris = itemgetter(*[str(track_id) for track_id in top_500_recommended_track_ids])(spotify_dataset.track_uri_map)

        line = ', '.join([pid] + [track_uri for track_uri in top_500_recommended_track_uris])
        lines.append(line)

    lines.append('\n')

    output_str = '\n'.join([line for line in lines])

    output_bytes = output_str.encode('utf-8')

    with gzip.open(filename, 'w') as fout:
        fout.write(output_bytes)  


# In[ ]:

r"""Recommend tracks for the test playlists.

Args:
    model : Recommender model.

Returns: 
    

"""

def test(model):

    test_edge_index = test_data.edge_index.to(device=args["device"])

    # Get the track nodes 
    track_nodes_mask = test_data["node_type"] == test_data["track_node_type"]
    track_node_index = torch.nonzero(track_nodes_mask).squeeze().to(args["device"])

    recommended_track_ids = torch.empty((0, args["k"]), dtype=torch.long)

    model.eval()

    # Split the test playlists in chunks
    playlist_index = test_data["test_nodes_index"]

    data_loader = playlist_index.chunk(args["test_chunk_size"])

    for step, batch_playlist_index in enumerate(data_loader):

        batch_playlist_index = batch_playlist_index.to(args["device"])

        out = model.recommend(edge_index=test_edge_index, src_index=batch_playlist_index, dst_index=track_node_index, k=args["k"]).cpu()

        recommended_track_ids = torch.concatenate([recommended_track_ids, out], dim=0)

    test_edge_index = test_edge_index.cpu()

    # Clear GPU memory
    torch.cuda.empty_cache()

    return recommended_track_ids


# In[ ]:

r"""Train recommender model.

Args:
    model : Recommender model.

Returns: 
    

"""
def train(model):
    
    test_edge_index = test_data.edge_index[:, train_data.num_edges:]

    tqdm_epoch_bar = trange(args["epochs"], desc='Training: ', leave=True)

    for epoch in tqdm_epoch_bar:

        model = model.to(device=args["device"])

        model.train()

        train_edge_index = train_data.edge_index.to(device=args["device"])

        playlist_index, pos_track_index, neg_track_index = structured_negative_sampling(train_data.edge_index, contains_neg_self_loops=False)

        # Shuffle the tensors
        shuffle_indices = torch.randperm(playlist_index.size()[0])
        playlist_index, pos_track_index, neg_track_index = playlist_index[shuffle_indices], pos_track_index[shuffle_indices], neg_track_index[shuffle_indices]

        data_loader = zip(playlist_index.chunk(args["train_chunk_size"]), pos_track_index.chunk(args["train_chunk_size"]), neg_track_index.chunk(args["train_chunk_size"]))
        
        for step, (batch_playlist_index, batch_pos_track_index, batch_neg_track_index) in enumerate(data_loader):

            batch_pos_edge_index = torch.stack([batch_playlist_index, batch_pos_track_index], dim=0)
            batch_neg_edge_index = torch.stack([batch_playlist_index, batch_neg_track_index], dim=0)

            batch_edge_index = torch.concatenate([batch_pos_edge_index, batch_neg_edge_index], dim=-1).to(device=args["device"])
            
            optimizer.zero_grad()

            out = model(edge_index=train_edge_index, edge_label_index=batch_edge_index)

            pos_edge_rank, neg_edge_rank = out.chunk(2)
            
            loss = model.recommendation_loss(pos_edge_rank, neg_edge_rank, lambda_reg=args["lambda_reg"])

            loss.backward()
            optimizer.step()

            tqdm_epoch_bar.set_description(f"Training: (Epoch - {epoch}), (Step - {step}), (BPRLoss - {loss.cpu().item()})")
            tqdm_epoch_bar.refresh()


        train_edge_index = train_edge_index.cpu()

        torch.cuda.empty_cache()

        # Get the recommended track ids for the test set
        recommended_track_ids = test(model)
        
        generate_output_file(recommended_track_ids, test_edge_index, filename=osp.join('output_files', f'epoch_{epoch}_submission.csv.gz'))


# In[ ]:

# Initiate the training
train(model)


