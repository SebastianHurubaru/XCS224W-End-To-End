import os
import os.path as osp
import shutil
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip


class SpotifyMPDataset(InMemoryDataset):

    name = 'spotify'

    def __init__(self, root, url, transform=None, pre_transform=None, pre_filter=None):
        
        self.url = url

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):

        return [
            'data.npz'
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):

        # Download and unpack the relevant data to `self.raw_dir` if it doesn't exist
        if os.path.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)):
            return
        
        for filename in self.raw_file_names:
            download_url(osp.join(f'{self.url}', f'{filename}'), self.raw_dir)

    def process(self):

        npz_data_file = np.load(osp.join(self.raw_dir, 'data.npz'))

        data = Data(
            x=torch.from_numpy(npz_data_file["x"]),
            edge_index=torch.from_numpy(npz_data_file["edge_index"]),
            edge_label=torch.from_numpy(npz_data_file["edge_label"]),
            train_edge_index=torch.from_numpy(npz_data_file["train_edge_index"]),
            test_edge_index=torch.from_numpy(npz_data_file["test_edge_index"])
        )

        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
