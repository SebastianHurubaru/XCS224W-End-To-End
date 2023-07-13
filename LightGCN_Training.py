#!/usr/bin/env python
# coding: utf-8

# # **(X)CS224W - End to End - Preprocessing**

# 

# In[ ]:


import os
import os.path as osp
from pathlib import Path

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from operator import itemgetter

import gzip

from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.models.lightgcn import LightGCN

from e2e.datasets import SpotifyMPDataset, EdgeDataset


# In[ ]:


spotify_dataset = SpotifyMPDataset(
    root='./spotify_mpd', 
    url=f"file://{osp.join(Path('.').resolve(), 'spotify_preprocessed_dataset')}"
)


# In[ ]:


train_data = spotify_dataset[0]
train_data


# In[ ]:


test_data = spotify_dataset[1]
test_data


# In[ ]:


device = torch.device("cpu")
if torch.cuda.is_available(): 
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")


# In[ ]:


args = {
    'device': device,
    'num_layers': 2,
    'hidden_dim': train_data.num_node_features,
    'batch_size': 2048,
    'dropout': 0.5,
    'lr': 0.001,
    'epochs': 10,
    'lambda_reg': 1e-4,
    'k': 600
}

args


# In[ ]:


model = LightGCN(
    num_nodes=test_data.num_nodes,
    embedding_dim=train_data.num_node_features,
    num_layers=3
)

# Initialize the embeddings with the initial features from the full graph
model.embedding.weight.data = test_data["x"]


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])


# In[ ]:


def generate_output_file(recommended_track_ids, edge_index, filename):

    lines = ['\n']

    team_line = "team_info, SH_XCS224W_Summer_2023, hurubaru@stanford.edu"

    lines.append(team_line)
    
    for i, playlist_node_id in enumerate(test_data["test_nodes_index"].numpy()):

        playlist_track_recommendations = list((recommended_track_ids[i]).numpy())

        excluded_track_nodes_mask = edge_index[0] == playlist_node_id

        playlist_excluded_tracks = edge_index[1, torch.nonzero(excluded_track_nodes_mask).squeeze()]

        valid_recommended_tracks = list(filter(lambda t: t not in playlist_excluded_tracks, playlist_track_recommendations))

        top_500_recommended_track_ids = valid_recommended_tracks[:500]

        pid = str(spotify_dataset.playlist_id_map.get(str(playlist_node_id)))

        top_500_recommended_track_uris = itemgetter(*[str(track_id) for track_id in top_500_recommended_track_ids])(spotify_dataset.track_uri_map)

        line = ', '.join([pid] + [track_uri for track_uri in top_500_recommended_track_uris])
        lines.append(line)

    lines.append('\n')

    output_str = '\n'.join([line for line in lines])

    output_bytes = output_str.encode('utf-8')

    with gzip.open(filename, 'w') as fout:
        fout.write(output_bytes)  


# In[ ]:


def test():

    test_edge_index = test_data.edge_index.to(device=args["device"])

    track_nodes_mask = test_data["node_type"] == test_data["track_node_type"]
    track_node_index = torch.nonzero(track_nodes_mask).squeeze().to(args["device"])

    recommended_track_ids = torch.empty((0, args["k"]), dtype=torch.long)

    tqdm_epoch_bar = trange(args["epochs"], desc='Evaluating: ', leave=True)
    for _ in tqdm_epoch_bar:

        model.eval()

        playlist_index = test_data["test_nodes_index"]

        edge_index_data = EdgeDataset(playlist_index)

        data_loader = DataLoader(edge_index_data, batch_size=args["batch_size"], shuffle=False, pin_memory=True, num_workers=0)
        
        for step, (batch_playlist_index, _, _) in enumerate(data_loader):

            out = model.recommend(edge_index=test_edge_index, src_index=batch_playlist_index, dst_index=track_node_index, k=args["k"]).cpu()

            recommended_track_ids = torch.concatenate([recommended_track_ids, out], dim=0)

            tqdm_epoch_bar.set_description(f"Evaluating: (Step - {step})")
            tqdm_epoch_bar.refresh()

    return recommended_track_ids


# In[ ]:


def train():
    
    model.to(device=args["device"])

    train_edge_index = train_data.edge_index.to(device=args["device"])

    test_edge_index = test_data.edge_index[:, train_data.num_edges:]

    tqdm_epoch_bar = trange(args["epochs"], desc='Training: ', leave=True)

    for epoch in tqdm_epoch_bar:

        model.train()

        playlist_index, pos_track_index, neg_track_index = structured_negative_sampling(train_data.edge_index, contains_neg_self_loops=False)

        edge_index_data = EdgeDataset(playlist_index, pos_track_index, neg_track_index)

        data_loader = DataLoader(edge_index_data, batch_size=args["batch_size"], shuffle=True, pin_memory=True, num_workers=0)
        
        for step, (batch_playlist_index, batch_pos_track_index, batch_neg_track_index) in enumerate(data_loader):

            batch_pos_edge_index = torch.stack([batch_playlist_index, batch_pos_track_index], dim=0)
            batch_neg_edge_index = torch.stack([batch_playlist_index, batch_neg_track_index], dim=0)

            batch_edge_index = torch.concatenate([batch_pos_edge_index, batch_neg_edge_index], dim=-1).to(device=args["device"])
            
            optimizer.zero_grad()

            out = model(edge_index=train_edge_index, edge_label_index=batch_edge_index).cpu()

            pos_edge_rank, neg_edge_rank = out.chunk(2)
            
            loss = model.recommendation_loss(pos_edge_rank, neg_edge_rank, lambda_reg=args["lambda_reg"])

            loss.backward()
            optimizer.step()

            tqdm_epoch_bar.set_description(f"Training: (Epoch - {epoch}), (BPRLoss - {loss.item()})")
            tqdm_epoch_bar.refresh()

    
        # Get the recommended track ids for the test set
        recommended_track_ids = test()
        
        generate_output_file(recommended_track_ids, test_edge_index, filename=osp.join('output_files', f'epoch_{epoch}_submission.csv.gz'))

train()


# In[ ]:






