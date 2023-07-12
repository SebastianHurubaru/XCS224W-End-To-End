import torch
import torch.nn.functional as F

# The PyG built-in GCNConv
from torch_geometric.nn import SGConv, ClusterGCNConv

class NeuralGraphCollaborativeFiltering(torch.nn.Module):
    """
    Neural Graph Collaborative Filtering
    """

    def __init__(self, node_emb, hidden_dim, num_layers, dropout):

        super(NeuralGraphCollaborativeFiltering, self).__init__()

        # Our nodes embeddings
        self.node_emb = node_emb

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(ClusterGCNConv(node_emb.embedding_dim, hidden_dim, add_self_loops=False, bias=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(
                SGConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(SGConv(hidden_dim, hidden_dim))

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    
    def forward(self, x_idx, edge_index):

        x = self.node_emb(x_idx)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.convs[-1](x, edge_index)

        return out