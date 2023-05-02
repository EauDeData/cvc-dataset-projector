from src.datautils.dataloaders import ZippedDataloader
from src.vision.models import Resnet, ResNetWithEmbedder, CLIPLoader
from src.projectors.projectors import PCAProjector, TSNEProjector
from src.methods.annoy import Annoyer

import cv2
import argparse
import torch
import pickle
import numpy as np
# https://open.spotify.com/track/6xE6ZWzK1YDDSYzqOCoQlz?si=b377b2524525413b



parser = argparse.ArgumentParser(
                    prog='Lorem Ipsum',
                    description='Super Trouper',
                    epilog='Uwu')

parser.add_argument('-f', '--file', default='example/windows.zip')      # option that takes a value
parser.add_argument('-pm', '--pretrained_model', default=None)      # option that takes a value
parser.add_argument('-m', '--model', default='clip')      # option that takes a value
parser.add_argument('-s', '--imsize', default = 224)

args = parser.parse_args()


if args.model.lower() == 'resnet': model = ResNetWithEmbedder(resnet = "18", embed_size = 256)
elif args.model.lower() == 'clip': model = CLIPLoader()
else: raise NotImplementedError
if not args.pretrained_model is None: 


    checkpoint = torch.load(args.pretrained_model)
    model.load_state_dict(checkpoint)



model.eval().cuda()
dataset = ZippedDataloader(args.file,)


annoyer = Annoyer(CLIPLoader().cuda(), dataset, 512)

annoyer.fit()

projector = TSNEProjector(dataset, model, imsize = args.imsize, mapsize = 20000)


print(
    "Using:\n",
    f"\tProjector: {projector}",
    f"\n\tModel: {model}"
    f"\n\tData: {args.file}"
)
image = cv2.cvtColor(projector.place_images().astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite('tmp.png', image)
