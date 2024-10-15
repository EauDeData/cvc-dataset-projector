import os
import annoy
import torch
import warnings

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class Annoyer:
    # High performance approaximate nearest neighbors - agnostic wrapper
    # Find implementation and documentation on https://github.com/spotify/annoy

    def __init__(self, model, dataset, emb_size=None, distance='angular', experiment_name='resnet_base', out_dir='/data3fast/users/amolina/output/', device='cuda') -> None:
        assert not (emb_size is None) and isinstance(emb_size, int),\
            f'When using Annoyer KNN emb_size must be an int. Set as None for common interface. Found: {type(emb_size)}'

        self.model = model

        # FIXME: Dataloader assumes 1 - Batch Size
        self.dataloader = dataset
        self.device = device

        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(
            out_dir, f'index.ann')

        self.trees = annoy.AnnoyIndex(emb_size, distance)
        self.state_variables = {
            'built': False,
        }

    def fit(self):
        if self.state_variables['built']:
            raise AssertionError('Cannot fit a built Annoy')
        else:
            self.state_variables['built'] = True
        
        for idx, image in enumerate(self.dataloader):
            print(
                f'Building KNN... {idx} / {len(self.dataloader)}\t', end='\r')

            with torch.no_grad():
                emb = self.model.infer(image).squeeze(
                )  # Ensure batch_size = 1
            self.trees.add_item(idx, emb)

        self.trees.build(10)  # 10 trees
        self.trees.save(self.path)

    def load(self):
        if self.state_variables['built']:
            raise AssertionError('Cannot load an already built Annoy')
        else:
            self.state_variables['built'] = True

        self.trees.load(self.path)

    def retrieve_by_idx(self, idx, n=50, **kwargs):
        return self.trees.get_nns_by_item(idx, n, **kwargs)

    def retrieve_by_vector(self, vector, n=50, **kwargs):
        return self.trees.get_nns_by_vector(vector, n, **kwargs)
