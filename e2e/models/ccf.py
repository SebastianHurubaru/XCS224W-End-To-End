import torch
import torch.nn.functional as F

class ConventionalCollaborativeFiltering(torch.nn.Module):
    """
    Conventional Collaborative Filtering
    """

    def __init__(self, node_emb):

        super(ConventionalCollaborativeFiltering, self).__init__()

        # Our nodes embeddings
        self.node_emb = node_emb
    
    def forward(self, edge_index):

        x_src = self.node_emb(edge_index[0])
        x_dst = self.node_emb(edge_index[1])

        out = torch.sum(x_src * x_dst, dim=-1)

        return out.squeeze()