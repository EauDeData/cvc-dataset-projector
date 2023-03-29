from PIL import Image
import numpy as np
import json
from sklearn.decomposition import PCA
import cv2


class BaseProjector:
    projector = None
    def __init__(self, dataset, model, mapsize = 20000, imsize = 75, prevent_overlap = True) -> None:
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

        for num, image in enumerate(self.dataset): self.embeddings[num] = self.model.infer(image) # Make Sure We can Infer Stuff
        return {x: y for x, y in enumerate(self.projector.fit_transform(list(self.embeddings.values())))}

    def place_images(self):

        xs, ys = [a[0] for a in self.projected.values()], [a[1] for a in self.projected.values()] # TODO: Don't iterate twice lol
        min_emb_x, max_emb_x, min_emb_y, max_emb_y = min(xs), max(xs), min(ys), max(ys)
        scale_coord = lambda var, min_, max_, scale: scale * (var - min_) / (max_ - min_)

        for num, image in enumerate(self.dataset):
            # FIXME: Test X,Y coords system 
            resized = cv2.resize(image, (self.imsize, self.imsize)) 
            x, y = self.projected[num]
            x_scaled, y_scaled = scale_coord(x, min_emb_x, max_emb_x, self.mapsize), scale_coord(y, min_emb_y, max_emb_y, self.mapsize)

            x_origin, y_origin = max(x_scaled - self.imsize, 0), max(y_scaled - self.imsize, 0)
            x_end, y_end = min(x_origin + self.imsize, self.mapsize - 1), min(y_origin + self.imsize, self.mapsize - 1)

            self.background[x_origin:x_end, y_origin:y_end, :] = resized
        
        return self.background

class PCAProjector(BaseProjector):
    

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.projector = PCA(2)

class TSNEProjector(BaseProjector):
    pass