from PIL import Image
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import cv2
from tqdm import tqdm
import torch

class BaseProjector:
    projector = None
    def __init__(self, dataset, model, mapsize = 2000, imsize = 75, prevent_overlap = True) -> None:
        # TODO: Use prevent Overlap

        self.dataset = dataset # Should Be An Iterable Of Images
        self.model = model # Should Be A Callable For Embeddings

        self.mapsize = mapsize
        self.imsize = imsize
        self.background = np.zeros((self.mapsize, self.mapsize, 3))

        self.embeddings = {

        }
        self.projected = self.fit_projector()

    def fit_projector(self):

        print('Projecting images...')
        for num, image in tqdm(enumerate(self.dataset)):
            with torch.no_grad():
                self.embeddings[num] = self.model.infer(image) # Make Sure We can Infer Stuff
                del image

        return {x: y for x, y in enumerate(self.projector.fit_transform(np.array(list(self.embeddings.values()))))}

    def place_images(self):

        xs, ys = [a[0] for a in self.projected.values()], [a[1] for a in self.projected.values()] # TODO: Don't iterate twice lol
        min_emb_x, max_emb_x, min_emb_y, max_emb_y = min(xs), max(xs), min(ys), max(ys)
        scale_coord = lambda var, min_, max_, scale: int(scale * (var - min_) / (max_ - min_))

        for num, image in enumerate(self.dataset):
            print(f'Drawing {num}...', end = '\r')
            # FIXME: Test X,Y coords system 
            resized = cv2.resize(image, (self.imsize, self.imsize)) 
            x, y = self.projected[num]
            x_scaled, y_scaled = scale_coord(x, min_emb_x, max_emb_x, self.mapsize), scale_coord(y, min_emb_y, max_emb_y, self.mapsize)

            x_origin, y_origin = max(x_scaled - self.imsize, 0), max(y_scaled - self.imsize, 0)
            x_end, y_end = min(x_origin + self.imsize, self.mapsize - 1), min(y_origin + self.imsize, self.mapsize - 1)

            round_error_x, round_error_y = self.imsize - (x_end - x_origin), self.imsize - (y_end - y_origin)
            self.background[x_origin:x_end + round_error_x, y_origin:y_end + round_error_y, :] = resized
        
        return self.background

class PCAProjector(BaseProjector):
    
    # I think this way of wrapping projectors is elegant
    #   But honeslty we could just pass the constructor
    def __init__(self, *args, **kwargs) -> None:
        self.projector = PCA(2)
        super().__init__(*args, **kwargs)

class TSNEProjector(BaseProjector):
    
    def __init__(self, *args, **kwargs) -> None:
        self.projector = TSNE(2)
        super().__init__(*args, **kwargs)