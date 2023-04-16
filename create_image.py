from src.datautils.dataloaders import ZippedDataloader
from src.vision.models import Resnet, ResNetWithEmbedder
from src.projectors.projectors import PCAProjector, TSNEProjector

import cv2
import argparse
import torch
# https://open.spotify.com/track/6xE6ZWzK1YDDSYzqOCoQlz?si=b377b2524525413b



parser = argparse.ArgumentParser(
                    prog='Lorem Ipsum',
                    description='Super Trouper',
                    epilog='Uwu')

parser.add_argument('-f', '--file', default='example/windows.zip')      # option that takes a value
parser.add_argument('-pm', '--pretrained_model', default=None)      # option that takes a value

args = parser.parse_args()

dataset = ZippedDataloader(args.file,)
model = ResNetWithEmbedder(resnet='18', embed_size=256, pretrained=False)
if not args.pretrained_model is None: model.load_state_dict(torch.load(args.pretrained_model))
model.eval()
projector = TSNEProjector(dataset, model, imsize = 128, mapsize = 20000)

image = projector.place_images()
cv2.imwrite('tmp.png', image)
