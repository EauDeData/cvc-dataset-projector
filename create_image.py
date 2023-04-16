from src.datautils.dataloaders import ZippedDataloader
from src.vision.models import Resnet
from src.projectors.projectors import PCAProjector, TSNEProjector

import cv2
import argparse

# https://open.spotify.com/track/6xE6ZWzK1YDDSYzqOCoQlz?si=b377b2524525413b



parser = argparse.ArgumentParser(
                    prog='Lorem Ipsum',
                    description='Super Trouper',
                    epilog='Uwu')

parser.add_argument('-f', '--file', default='example/windows.zip')      # option that takes a value
parser.add_argument('-pm', '--pretrained_model', default=None)      # option that takes a value

args = parser.parse_args()

dataset = ZippedDataloader(args.file,)
model = Resnet(resnet='101')
if not args.pretrained_model is None: model = model.load(args.pretrained_model)
projector = TSNEProjector(dataset, model, imsize = 128, mapsize = 20000)

image = projector.place_images()
cv2.imwrite('tmp.png', image)
