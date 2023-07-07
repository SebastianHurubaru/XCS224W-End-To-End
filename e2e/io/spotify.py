import os
import os.path as osp
import json
import re
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from sklearn.decomposition import PCA

from torch_geometric.data import HeteroData
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, one_hot, remove_self_loops

from sentence_transformers import SentenceTransformer

def read_spotify_data(folder, files, device):

    model = SentenceTransformer('all-MiniLM-L6-v2').to(device=device)
    model.eval()

    print(f'Running SentenceTransformer model on device: {device}')

    track_by_artist_edge_index = np.empty((2, 0), dtype=np.long)
    track_on_album_edge_index = np.empty((2, 0), dtype=np.long)
    album_by_artist_edge_index = np.empty((2, 0), dtype=np.long)
    playlist_contains_track_edge_index = np.empty((2, 0), dtype=np.long)

    playlist_contains_track_edge_label = np.empty(0, dtype=np.long)

    node_emb_dim = 32
    pca = PCA(n_components=node_emb_dim)
    
    x_artist = np.empty((0, node_emb_dim))
    x_track = np.empty((0, node_emb_dim))
    x_album = np.empty((0, node_emb_dim))
    x_playlist = np.empty((0, node_emb_dim))

    current_playlist_node_id = 0

    artist_ids = defaultdict()
    tracks_ids = defaultdict()
    album_ids = defaultdict()

    def read_and_process_files(files):

        nonlocal track_by_artist_edge_index
        nonlocal track_on_album_edge_index
        nonlocal album_by_artist_edge_index
        nonlocal playlist_contains_track_edge_index
        nonlocal playlist_contains_track_edge_label

        nonlocal x_artist
        nonlocal x_track
        nonlocal x_album
        nonlocal x_playlist
        nonlocal current_playlist_node_id

        nonlocal artist_ids
        nonlocal tracks_ids
        nonlocal album_ids

        for filename in tqdm(files, position=0, desc="File processing"):
            
            fullpath = os.sep.join((folder, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            
            current_artist_attributes = []
            current_track_attributes = []
            current_album_attributes = []
            current_playlist_attributes = []

            current_track_by_artist_edge_index = []
            current_track_on_album_edge_index = []
            current_album_by_artist_edge_index = []
            current_playlist_contains_track_edge_index = []

            current_playlist_contains_track_edge_label = []

            for playlist in mpd_slice.get("playlists", []):
                
                playlist_id = current_playlist_node_id; current_playlist_node_id += 1
            
                current_playlist_attributes.append(normalize_name(playlist.get("name", "Unknown")))

                for track_item in playlist.get("tracks", []):

                    artist, added_artist = update_item_attributes_list(artist_ids, track_item.get("artist_uri"), track_item.get("artist_name"), current_artist_attributes)
                    track, added_track = update_item_attributes_list(tracks_ids, track_item.get("track_uri"), track_item.get("track_name"), current_track_attributes)
                    album, added_album = update_item_attributes_list(album_ids, track_item.get("album_uri"), track_item.get("album_name"), current_album_attributes)

                    
                    if added_track == True or added_artist == True:
                        current_track_by_artist_edge_index.append([track.get("idx"), artist.get("idx")])

                    if added_track == True or added_album == True:
                        current_track_on_album_edge_index.append([track.get("idx"), album.get("idx")])
                    
                    if added_album == True or added_artist == True:
                        current_album_by_artist_edge_index.append([album.get("idx"), artist.get("idx")])

                    current_playlist_contains_track_edge_index.append([playlist_id, track.get("idx")])
                    current_playlist_contains_track_edge_label.append(track_item.get("pos"))

            total_emb = pca.fit_transform(
                    model.encode(
                        current_artist_attributes+current_track_attributes+current_album_attributes+current_playlist_attributes,
                        show_progress_bar=True
                    )
                )
            
            total_emb = torch.randn(len(current_artist_attributes+current_track_attributes+current_album_attributes+current_playlist_attributes), node_emb_dim)

            slice_index = 0
            
            artist_emb = total_emb[:slice_index+len(current_artist_attributes)]
            slice_index += len(current_artist_attributes)

            track_emb = total_emb[slice_index:slice_index+len(current_track_attributes)]
            slice_index += len(current_track_attributes)

            album_emb = total_emb[slice_index:slice_index+len(current_album_attributes)]
            slice_index += len(current_album_attributes)

            playlist_emb = total_emb[slice_index:slice_index+len(current_playlist_attributes)]
            slice_index += len(current_playlist_attributes)
            
            x_artist = np.concatenate((x_artist, artist_emb), axis=0)
            x_track = np.concatenate((x_track, track_emb), axis=0)
            x_album = np.concatenate((x_album, album_emb), axis=0)
            x_playlist = np.concatenate((x_playlist, playlist_emb), axis=0)

            if len(current_track_by_artist_edge_index) > 0:
                track_by_artist_edge_index = np.concatenate((track_by_artist_edge_index, torch.LongTensor(current_track_by_artist_edge_index).T), axis=-1)

            if len(current_track_on_album_edge_index) > 0:
                track_on_album_edge_index = np.concatenate((track_on_album_edge_index, torch.LongTensor(current_track_on_album_edge_index).T), axis=-1)

            if len(current_album_by_artist_edge_index) > 0:  
                album_by_artist_edge_index = np.concatenate((album_by_artist_edge_index, torch.LongTensor(current_album_by_artist_edge_index).T), axis=-1)
            
            if len(current_playlist_contains_track_edge_index) > 0:
                playlist_contains_track_edge_index = np.concatenate((playlist_contains_track_edge_index, torch.LongTensor(current_playlist_contains_track_edge_index).T), axis=-1)
            
            if len(current_playlist_contains_track_edge_label) > 0:
                playlist_contains_track_edge_label = np.concatenate((playlist_contains_track_edge_label, torch.LongTensor(current_playlist_contains_track_edge_label)), axis=0)

    # Create the graph data out of the Spotify MPD
    read_and_process_files(files[:-1])

    start_test_playlist_contains_track_edge_index = playlist_contains_track_edge_index.shape[0]

    # Append the test graph data from the spotify challenge
    read_and_process_files([files[-1]])

    data = {
            'artist': { 
                'x': x_artist 
            },
            'track': { 
                'x': x_track 
            },
            'album': { 
                'x': x_album 
            },
            'playlist': { 
                'x': x_playlist 
            },
            ('track', 'by', 'artist'): { 
                'edge_index': track_by_artist_edge_index 
            },
            ('track', 'on', 'album'): { 
                'edge_index': track_on_album_edge_index 
            },
            ('album', 'by', 'artist'): { 
                'edge_index': album_by_artist_edge_index 
            },
            ('playlist', 'contains', 'track'): { 
                'edge_index': playlist_contains_track_edge_index,
                'edge_label': playlist_contains_track_edge_label,
                'test_edge_index': range(start_test_playlist_contains_track_edge_index, playlist_contains_track_edge_index.shape[0])
            },
        }
    
    return data

def update_item_attributes_list(ids, item_uri, item_name, item_attributes):
    
    added = False

    existing_item = ids.get(item_uri, None)

    if existing_item is not None:
        item = existing_item
    else:
        item = {
            "idx": len(ids),
            "name": item_name
        }
        ids.update({item_uri: item})
        item_attributes.append(item_name)
        added = True

    return item, added

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name
