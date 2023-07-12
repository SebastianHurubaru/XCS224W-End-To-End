
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


from e2e.datasets import SpotifyMPDataset, EdgeDataset
from e2e.models import NeuralGraphCollaborativeFiltering, ConventionalCollaborativeFiltering



spotify_dataset = SpotifyMPDataset(
    root='./spotify_mpd', 
    url=f"file://{osp.join(Path('.').resolve(), 'spotify_preprocessed_dataset')}",
    pre_transform=ToUndirected(reduce="mean")
)

train_data = spotify_dataset[0]
test_data = spotify_dataset[1]

train_data.validate(raise_on_error=True)

print(f'Undirected = {train_data.is_undirected()}')
# print(f'Bipartite = {data[("playlist", "contains", "track")].is_bipartite()}')
              
emb = nn.Embedding(num_embeddings=test_data.num_nodes, embedding_dim=train_data.num_node_features)
emb.weight.data = test_data["x"]

device = torch.device("cpu")
if torch.cuda.is_available(): 
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")

# link_split_transform = RandomLinkSplit(
#     num_val=0.05,
#     num_test=0,
#     is_undirected=True, 
#     key="edge_label", 
#     # edge_types=[('playlist', 'contains', 'track')],
#     # rev_edge_types=[('playlist', 'contains', 'track')],
#     split_labels=True, 
#     add_negative_train_samples=True,
#     disjoint_train_ratio=0.2
# )

# model = NGCF(
#     node_emb=emb,
#     hidden_dim=32,
#     num_layers=2,
#     dropout=0.2
# ).to(device)

link_split_transform = RandomLinkSplit(
    num_val=0,
    num_test=0,
    is_undirected=True, 
    key="edge_label", 
    split_labels=False, 
    add_negative_train_samples=True,
    neg_sampling_ratio=1,
    disjoint_train_ratio=0
)

model = ConventionalCollaborativeFiltering(
    node_emb=emb
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