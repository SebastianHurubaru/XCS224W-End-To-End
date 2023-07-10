import json
import os
import os.path as osp
import re
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, one_hot, remove_self_loops
from tqdm import tqdm


def read_spotify_data(folder, files, device):

    model = SentenceTransformer('all-MiniLM-L6-v2').to(device=device)
    model.eval()

    print(f'Running SentenceTransformer model on device: {device}')

    edge_index = np.empty((2, 0), dtype=np.long)
    edge_label = np.empty(0, dtype=np.long)

    node_emb_dim = 32
    pca = PCA(n_components=node_emb_dim)
    
    x = np.empty((0, node_emb_dim)) 

    current_node_id = 0
    tracks_ids = defaultdict()

    def read_and_process_files(files):

        nonlocal edge_index
        nonlocal edge_label

        nonlocal x

        nonlocal current_node_id
        nonlocal tracks_ids

        for filename in tqdm(files, position=0, desc="File processing"):
            
            fullpath = os.sep.join((folder, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            
            current_node_attributes = []

            current_edge_index = []
            current_edge_label = []

            for playlist in mpd_slice.get("playlists", []):
                
                playlist_id = current_node_id; current_node_id += 1

                if "name" in playlist:
                    playlist_name = normalize_name(f'{playlist.get("name")}')    
                    current_node_attributes.append(f'This is a music playlist named "{playlist_name}".')
                else:
                    current_node_attributes.append(f'This is an unnamed music playlist.')

                for track_item in playlist.get("tracks", []):

                    track, current_node_id = update_item_attributes_list(tracks_ids, current_node_id, track_item, current_node_attributes)
                    
                    current_edge_index.append([playlist_id, track.get("idx")])
                    current_edge_label.append(track_item.get("pos"))

            # current_node_emb = pca.fit_transform(
            #     model.encode(
            #         current_node_attributes,
            #         show_progress_bar=True
            #     )
            # )
            
            current_node_emb = torch.randn(len(current_node_attributes), node_emb_dim)

            x = np.concatenate((x, current_node_emb), axis=0)
            
            if len(current_edge_index) > 0:
                edge_index = np.concatenate((edge_index, torch.LongTensor(current_edge_index).T), axis=-1)
            
            if len(current_edge_label) > 0:
                edge_label = np.concatenate((edge_label, torch.LongTensor(current_edge_label)), axis=0)

    # Create the graph data out of the Spotify MPD
    read_and_process_files(files[:-1])

    start_test_edge_index = edge_index.shape[1]

    # Append the test graph data from the spotify challenge
    read_and_process_files([files[-1]])

    data = {
        'x': x,
        'edge_index': edge_index,
        'edge_label': edge_label,
        'train_edge_index': edge_index[:, :start_test_edge_index],
        'test_edge_index': edge_index[:, start_test_edge_index:edge_index.shape[1]]
    }
    
    return data

def update_item_attributes_list(ids, current_node_id, playlist_track_item, item_attributes):

    existing_item = ids.get(playlist_track_item.get("track_uri"), None)

    if existing_item is not None:
        item = existing_item
    else:
        item = {
            "idx": current_node_id
        }
        current_node_id += 1
        ids.update({playlist_track_item.get("track_uri"): item})
        item_attributes.append(
            f'The music track "{playlist_track_item.get("track_name")}" by {playlist_track_item.get("artist_name")} was released on "{playlist_track_item.get("album_name")}" album with a duration of {playlist_track_item.get("duration_ms")} milliseconds.'
        )

    return item, current_node_id

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name
