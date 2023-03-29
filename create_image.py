from src.datautils.dataloaders import ZippedDataloader
from src.vision.models import Resnet
from src.projectors.projectors import PCAProjector

import cv2

dataset = ZippedDataloader('/home/adri/Pictures/zipper.zip')
model = Resnet(resnet='50')
projector = PCAProjector(dataset, model)

image = projector.place_images()
cv2.imwrite('tmp.png', image)