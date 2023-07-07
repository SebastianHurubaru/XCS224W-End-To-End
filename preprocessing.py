import os
import os.path as osp
import shutil

import numpy as np

import argparse
from pathlib import Path

import pandas as pd

import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_zip

from e2e.io import read_spotify_data

dataset_name = 'spotify_million_playlist_dataset'
challenge_name = 'spotify_million_playlist_dataset_challenge'

raw_file_names = []
for i in range(0, 10**3, 10**3):
    raw_file_names.append(f'mpd.slice.{i}-{i+999}.json')

def get_raw_file_names():

    raw_file_names = []
    
    # Add the milion playlist dataset split files
    for i in range(0, 10**6, 10**3):
        raw_file_names.append(f'mpd.slice.{i}-{i+999}.json')
    
    # Add the million playlist challenge file
    raw_file_names.append('challenge_set.json')

    return raw_file_names

def main(args):

    device = torch.device("cpu")
    if (args.device == 'gpu'):
        if torch.cuda.is_available(): 
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")

    output_dataset_path = args.output_dataset_path if args.output_dataset_path.is_absolute() else osp.join(Path('.').resolve(), args.output_dataset_path)
    orig_dataset_path = args.orig_dataset_path if args.orig_dataset_path.is_absolute() else osp.join(Path('.').resolve(), args.orig_dataset_path)

    raw_dir_path = osp.join(output_dataset_path, 'raw')

    raw_file_names = get_raw_file_names()

    download(raw_dir_path, orig_dataset_path, output_dataset_path, raw_file_names)
    process(raw_dir_path, output_dataset_path, device, raw_file_names)


def download(raw_dir_path, orig_dataset_path, output_dataset_path, raw_file_names):
        
    # Download and unpack the relevant data to `raw_dir_path` if it doesn't exist
    if os.path.isdir(raw_dir_path) and len(os.listdir(raw_dir_path)) > 0:
        return
    
    dataset_folder = osp.join(output_dataset_path, dataset_name)
    dataset_path = download_url(f'file://{osp.join(orig_dataset_path, dataset_name)}.zip', dataset_folder)
    extract_zip(dataset_path, dataset_folder)
    os.unlink(dataset_path)

    challenge_folder = osp.join(output_dataset_path, challenge_name)
    challenge_path = download_url(f'file://{osp.join(orig_dataset_path, challenge_name)}.zip', dataset_folder)
    extract_zip(challenge_path, challenge_folder)
    os.unlink(challenge_path)

    # Rename the data subfolder of the dataset as the raw dir
    os.rename(osp.join(f'{dataset_folder}', 'data'), raw_dir_path)

    # Copy the challenge file to the raw dir
    shutil.copyfile(osp.join(f'{challenge_folder}', f'{raw_file_names[-1]}'), osp.join(f'{raw_dir_path}', f'{raw_file_names[-1]}'))

    # Remove the intermediate folders
    shutil.rmtree(dataset_folder)
    shutil.rmtree(challenge_folder)

def process(raw_dir_path, output_dir_path, device, raw_file_names):

    spotify_data = read_spotify_data(raw_dir_path, raw_file_names, device)

    np.savez(osp.join(output_dir_path, 'artist.npz'), x=spotify_data["artist"]["x"])
    np.savez(osp.join(output_dir_path, 'track.npz'), x=spotify_data["track"]["x"])
    np.savez(osp.join(output_dir_path, 'album.npz'), x=spotify_data["album"]["x"])
    np.savez(osp.join(output_dir_path, 'playlist.npz'), x=spotify_data["playlist"]["x"])

    np.savez(osp.join(output_dir_path, 'track_by_artist.npz'), edge_index=spotify_data[('track', 'by', 'artist')]['edge_index'])
    np.savez(osp.join(output_dir_path, 'track_on_album.npz'), edge_index=spotify_data[('track', 'on', 'album')]['edge_index'])
    np.savez(osp.join(output_dir_path, 'album_by_artist.npz'), edge_index=spotify_data[('album', 'by', 'artist')]['edge_index'])
    np.savez(
        osp.join(output_dir_path, 'playlist_contains_track.npz'), 
        edge_index=spotify_data[('playlist', 'contains', 'track')]['edge_index'],
        edge_label=spotify_data[('playlist', 'contains', 'track')]['edge_label'],
        test_edge_index=spotify_data[('playlist', 'contains', 'track')]['test_edge_index']
    )    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pre-process the Spotify Million Playlists Dataset.')

    parser.add_argument('--orig_dataset_path', type=Path, default='./spotify_raw_dataset', required=False,
                        help='Path to original Spotify MPD')
    
    parser.add_argument('--output_dataset_path', type=Path, default='./spotify_preprocessed_dataset', required=False,
                        help='Path to the output pre-processed Spotify MPD')
    
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], required=False,
                        help='Device to be used')
    
    args = parser.parse_args()

    main(args)