from src.datautils.dataloaders import ZippedDataloader
from src.vision.models import Resnet
from src.projectors.projectors import PCAProjector

import cv2

# https://open.spotify.com/track/6xE6ZWzK1YDDSYzqOCoQlz?si=b377b2524525413b

dataset = ZippedDataloader('example/windows.zip')
model = Resnet(resnet='50')
projector = PCAProjector(dataset, model, imsize = 30, mapsize = 300)

image = projector.place_images()
cv2.imwrite('tmp.png', image)