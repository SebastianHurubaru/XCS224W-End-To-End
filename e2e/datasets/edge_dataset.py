from torch.utils.data import Dataset

class EdgeDataset(Dataset):
    def __init__(self, edge_index, edge_label):
        self.edge_index = edge_index
        self.edge_label = edge_label

    def __len__(self):
        return self.edge_index.shape[-1]

    def __getitem__(self, idx):
        item = self.edge_index[:, idx]
        label = self.edge_label[idx]
        return item, label