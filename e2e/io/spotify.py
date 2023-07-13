import json
import os
import os.path as osp
import re
from collections import OrderedDict

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

    train_edge_index = np.empty((2, 0), dtype=np.long)
    train_edge_label = np.empty(0, dtype=np.long)

    test_edge_index = np.empty((2, 0), dtype=np.long)
    test_edge_label = np.empty(0, dtype=np.long)

    node_emb_dim = 32
    pca = PCA(n_components=node_emb_dim)
    
    x_playlist_train = np.empty((0, node_emb_dim), dtype=np.float32)
    x_playlist_test = np.empty((0, node_emb_dim), dtype=np.float32)

    x_track = np.empty((0, node_emb_dim), dtype=np.float32)

    playlist_node_ids = OrderedDict()
    tracks_node_ids = OrderedDict()

    def read_and_process_files(files, x_playlist, edge_index, edge_label):

        nonlocal playlist_node_ids
        nonlocal tracks_node_ids

        nonlocal x_track

        for filename in tqdm(files, position=0, desc="File processing"):
            
            fullpath = os.sep.join((folder, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            
            current_playlist_features = []
            current_tracks_features = []

            current_edge_index = []
            current_edge_label = []

            for playlist in mpd_slice.get("playlists", []):
                
                playlist_node_id = len(playlist_node_ids)
                playlist_node_ids.update({playlist.get("pid"): playlist_node_id})

                if "name" in playlist:
                    playlist_name = normalize_name(f'{playlist.get("name")}')    
                    current_playlist_features.append(f'This is a music playlist named "{playlist_name}".')
                else:
                    current_playlist_features.append(f'This is an unnamed music playlist.')

                for track_item in playlist.get("tracks", []):

                    track_node_id = tracks_node_ids.get(track_item.get("track_uri"), -1)

                    if track_node_id == -1:
                        track_node_id = len(tracks_node_ids)
                        tracks_node_ids.update({track_item.get("track_uri"): track_node_id})
                        current_tracks_features.append(
                            f'The music track "{track_item.get("track_name")}" by {track_item.get("artist_name")} was released on "{track_item.get("album_name")}" album with a duration of {track_item.get("duration_ms")} milliseconds.'
                        )
                    
                    current_edge_index.append([playlist_node_id, track_node_id])
                    current_edge_label.append(track_item.get("pos"))

            # current_node_emb = pca.fit_transform(
            #     model.encode(
            #         current_playlist_features+current_tracks_features,
            #         show_progress_bar=True
            #     )
            # )
            
            current_node_emb = torch.randn(len(current_playlist_features)+len(current_tracks_features), node_emb_dim)

            slice_index = 0
            
            current_playlist_emb = current_node_emb[slice_index:slice_index+len(current_playlist_features)]
            slice_index += len(current_playlist_features)

            current_track_emb = current_node_emb[slice_index:slice_index+len(current_tracks_features)]
            slice_index += len(current_tracks_features)

            x_playlist = np.concatenate((x_playlist, current_playlist_emb), axis=0)
            x_track = np.concatenate((x_track, current_track_emb), axis=0)
            
            if len(current_edge_index) > 0:
                edge_index = np.concatenate((edge_index, torch.LongTensor(current_edge_index).T), axis=-1)
            
            if len(current_edge_label) > 0:
                edge_label = np.concatenate((edge_label, torch.LongTensor(current_edge_label)), axis=0)

        return x_playlist, edge_index, edge_label

    # Create the graph data out of the Spotify MPD
    x_playlist_train, train_edge_index, train_edge_label = read_and_process_files(files[:-1], x_playlist=x_playlist_train, edge_index=train_edge_index, edge_label=train_edge_label)

    # Append the test graph data from the spotify challenge
    x_playlist_test, test_edge_index, test_edge_label = read_and_process_files([files[-1]], x_playlist=x_playlist_test, edge_index=test_edge_index, edge_label=test_edge_label)

    # Do some validation
    pos = 0
    for _, playlist_node_id in playlist_node_ids.items():
        assert pos == playlist_node_id, "Playlist node ids do not match!"
        pos += 1

    pos = 0
    for _, track_node_id in tracks_node_ids.items():
        assert pos == track_node_id, "Track node ids do not match!"
        pos += 1

    data = {
        'x_playlist_train': x_playlist_train,
        'x_playlist_test': x_playlist_test,
        'x_track': x_track,
        'train_edge_index': train_edge_index,
        'train_edge_label': train_edge_label,
        'test_edge_index': test_edge_index,
        'test_edge_label': test_edge_label,
        'playlist_id_map': {v: k for k, v in playlist_node_ids.items()},
        'track_uri_map': {v: k for k, v in tracks_node_ids.items()}
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
