import os
import os.path as osp
import shutil
import numpy as np

import gzip
import json

from collections import OrderedDict

import torch
from torch_geometric.data import InMemoryDataset, HeteroData, download_url, extract_zip

class SpotifyMPDataset(InMemoryDataset):

    name = 'spotify'

    def __init__(self, root, url, transform=None, pre_transform=None, pre_filter=None):
        
        self.url = url

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):

        return [
            'data.npz',
            'playlist_id_map.json.gz',
            'track_uri_map.json.gz'
        ]

    @property
    def processed_file_names(self):
        return ['data.pt', 'playlist_id_map.json.gz', 'track_uri_map.json.gz']

    def download(self):

        # Download and unpack the relevant data to `self.raw_dir` if it doesn't exist
        if os.path.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)):
            return
        
        for filename in self.raw_file_names:
            download_url(osp.join(f'{self.url}', f'{filename}'), self.raw_dir)

    def process(self):

        npz_data_file = np.load(osp.join(self.raw_dir, 'data.npz'))

        train_data = HeteroData(
            track={ 'x': torch.from_numpy(npz_data_file["x_track"]) },
            playlist={ 'x': torch.from_numpy(npz_data_file["x_playlist_train"]) },
            playlist__contains__track={ 
                'edge_index': torch.from_numpy(npz_data_file["train_edge_index"]),
                'edge_label': torch.from_numpy(npz_data_file["train_edge_label"])
            }
        ).to_homogeneous()

        
        """
        Node type new mapping after converting to a homogeneuous graph: {
            "playlist": 1,
            "track": 0
        }
        """

        train_data["track_node_type"] = 0
        train_data["playlist_node_type"] = 1

        test_data = HeteroData(
            track={ 'x': torch.from_numpy(npz_data_file["x_track"]) },
            playlist={ 'x': torch.from_numpy(np.concatenate((npz_data_file["x_playlist_train"], npz_data_file["x_playlist_test"]), axis=0)) },
            playlist__contains__track={ 
                'edge_index': torch.from_numpy(np.concatenate((npz_data_file["train_edge_index"], npz_data_file["test_edge_index"]), axis=-1)),
                'edge_label': torch.from_numpy(np.concatenate((npz_data_file["train_edge_label"], npz_data_file["test_edge_label"]), axis=0))
            }
        ).to_homogeneous()

        test_data["track_node_type"] = 0
        test_data["playlist_node_type"] = 1
        test_data["test_nodes_index"] = torch.arange(train_data.num_nodes, test_data.num_nodes)

        data_list = [train_data, test_data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Make sure the edge attributes/labels are integers
        for data in data_list:
            data["edge_label"] = data["edge_label"].to(dtype=torch.long)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        shutil.copyfile(osp.join(self.raw_dir, 'playlist_id_map.json.gz'), self.processed_paths[1])
        shutil.copyfile(osp.join(self.raw_dir, 'track_uri_map.json.gz'), self.processed_paths[2])

    def read_compressed_json_file_to_dict(self, file):
        with gzip.open(file, 'r') as fin:
            json_bytes = fin.read()

        json_str = json_bytes.decode('utf-8')
        return json.loads(json_str, type=OrderedDict)
