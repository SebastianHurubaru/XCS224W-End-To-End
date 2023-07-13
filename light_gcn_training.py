
import os
import os.path as osp
from pathlib import Path

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.models.lightgcn import LightGCN

from e2e.datasets import SpotifyMPDataset, EdgeDataset



spotify_dataset = SpotifyMPDataset(
    root='./spotify_mpd', 
    url=f"file://{osp.join(Path('.').resolve(), 'spotify_preprocessed_dataset')}"
)

train_data = spotify_dataset[0]
test_data = spotify_dataset[1]

train_data.validate(raise_on_error=True)

print(f'Undirected = {train_data.is_undirected()}')
              
emb = nn.Embedding(num_embeddings=test_data.num_nodes, embedding_dim=train_data.num_node_features)
emb.weight.data = test_data["x"]

device = torch.device("cpu")
if torch.cuda.is_available(): 
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")

model = LightGCN(
    num_nodes=train_data.num_nodes,
    embedding_dim=train_data.num_node_features,
    num_layers=3
).to(device)

epochs = 500
batch_size = 10**4
learning_rate = 0.2

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

tqdm_epoch_bar = trange(epochs, desc='Training: ', leave=True)

model.train()

for _ in tqdm_epoch_bar:

    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        force_undirected=train_data.is_undirected())

    edge_label = 500 - train_data.edge_label
    edge_index = torch.concatenate((train_data.edge_index, neg_edge_index), dim=-1)
    edge_label = torch.concatenate((train_data.edge_label, torch.full((neg_edge_index.shape[-1],), -1)), dim=0)

    edge_index_data = EdgeDataset(edge_index, edge_label)

    tqdm_batch_bar = tqdm(DataLoader(edge_index_data, batch_size=batch_size, shuffle=True, num_workers=0))
    
    for step, (batch_edge_index, batch_edge_label) in enumerate(tqdm_batch_bar):

        batch_edge_index = batch_edge_index.T.to(device)

        optimizer.zero_grad()

        pred = model(batch_edge_index).cpu()
        
        loss = torch.nn.functional.mse_loss(pred, batch_edge_label.to(torch.float32))

        loss.backward()
        optimizer.step()

        tqdm_epoch_bar.set_description(f"Training: (mse_loss - {loss.item()})")
        tqdm_epoch_bar.refresh()

        tqdm_batch_bar.set_description(f"Iteration: {step})")
        tqdm_batch_bar.refresh()

print("Finish")