import argparse
import os

import h5py
import mindspore as ms
import numpy as np
from tqdm import tqdm

from kmeans import KMeans


def feature_based_clustering(wsi_h5, radius=1):
    total_coords, total_features = np.array(wsi_h5['coords'], dtype=np.float32), np.array(wsi_h5['features'], dtype=np.float32)
    print(total_coords.shape, total_features.shape)
    assert total_coords.shape[0] == total_features.shape[0]
    N_clusters = 256 # number of clusters

    cuda_coords = ms.Tensor(total_features, dtype=ms.float32)
    kmeans = KMeans(n_clusters=N_clusters, mode='euclidean')
    kmeans.fit(cuda_coords)
    cluster_labels = kmeans.predict(cuda_coords).asnumpy()
    cluster_data = dict()

    cluster_data['cluster_labels'] = cluster_labels

    return cluster_data

def createDir_h5(h5_path, save_path):
    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        pbar.set_description('%s - Creating Graph' % (h5_fname[:12]))
        save_fname = os.path.join(save_path, h5_fname[:-3] + '.h5')
        try:
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            G = feature_based_clustering(wsi_h5)
            # Save the result into a new H5 file
            with h5py.File(save_fname, 'w') as hf:
                for key, value in G.items():
                    hf.create_dataset(key, data=value)
            
            wsi_h5.close()
        except OSError:
            pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--h5_path', type = str,
					help='path to folder containing coordinates stored in .h5 files')
parser.add_argument('--save_path', type = str,
					help='path to folder will be used to save clusters')

if __name__ == '__main__': 
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    createDir_h5(args.h5_path, args.save_path)

    for h5_file in os.listdir(args.save_path):
        h5_file = os.path.join(args.save_path, h5_file)
        with h5py.File(h5_file, 'r') as hf:
            print(hf.keys())
            print(hf['cluster_labels'].shape)
            print(hf['cluster_labels'][:10])
    