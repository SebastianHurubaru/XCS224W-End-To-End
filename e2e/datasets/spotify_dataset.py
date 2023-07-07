import os
import os.path as osp
import shutil
import numpy as np

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
            'artist.npz',
            'track.npz',
            'album.npz',
            'playlist.npz',
            'track_by_artist.npz',
            'track_on_album.npz',
            'album_by_artist.npz',
            'playlist_contains_track.npz'
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

        artist = np.load(osp.join(self.raw_dir, 'artist.npz'))
        track = np.load(osp.join(self.raw_dir, 'track.npz'))
        album = np.load(osp.join(self.raw_dir, 'album.npz'))
        playlist = np.load(osp.join(self.raw_dir, 'playlist.npz'))

        track_by_artist = np.load(osp.join(self.raw_dir, 'track_by_artist.npz'))
        track_on_album = np.load(osp.join(self.raw_dir, 'track_on_album.npz'))
        album_by_artist = np.load(osp.join(self.raw_dir, 'album_by_artist.npz'))
        playlist_contains_track = np.load(osp.join(self.raw_dir, 'playlist_contains_track.npz'))

        data = HeteroData()

        data['artist'].x = torch.from_numpy(artist['x'])
        data['track'].x = torch.from_numpy(track['x'])
        data['album'].x = torch.from_numpy(album['x'])
        data['playlist'].x = torch.from_numpy(playlist['x'])

        data['track', 'by', 'artist'].edge_index = torch.from_numpy(track_by_artist['edge_index']).to(torch.long)
        data['track', 'on', 'album'].edge_index = torch.from_numpy(track_on_album['edge_index']).to(torch.long)
        data['album', 'by', 'artist'].edge_index = torch.from_numpy(album_by_artist['edge_index']).to(torch.long)
        data['playlist', 'contains', 'track'].edge_index = torch.from_numpy(playlist_contains_track['edge_index']).to(torch.long)
        data['playlist', 'contains', 'track'].edge_label = torch.from_numpy(playlist_contains_track['edge_label']).to(torch.long)
        data['playlist', 'contains', 'track'].test_edge_index = torch.from_numpy(playlist_contains_track['test_edge_index']).to(torch.long)

        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
