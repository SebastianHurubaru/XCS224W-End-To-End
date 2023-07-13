from torch.utils.data import Dataset

class EdgeDataset(Dataset):
    def __init__(self, src_node_index, pos_dst_node_index=None, neg_dst_node_index=None):
        self.src_node_index = src_node_index
        self.pos_dst_node_index = pos_dst_node_index
        self.neg_dst_node_index = neg_dst_node_index

    def __len__(self):
        return self.src_node_index.shape[-1]

    def __getitem__(self, idx):
        return (
            self.src_node_index[idx], 
            -1 if self.pos_dst_node_index == None else self.pos_dst_node_index[idx], 
            -1 if self.neg_dst_node_index == None else self.neg_dst_node_index[idx]
        )