from src.datautils.dataloaders import ZippedDataloader
from src.vision.models import Resnet, ResNetWithEmbedder, CLIPLoader, CLIPTextEncoder, SymmetricSiameseModel
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
parser.add_argument('-m', '--model', default='clip')      # option that takes a value
parser.add_argument('-s', '--imsize', default = 224)

args = parser.parse_args()


if args.model.lower() == 'resnet': model = ResNetWithEmbedder(resnet = "18", embed_size = 256)
elif args.model.lower() == 'clip': model = CLIPLoader()
else: raise NotImplementedError
if not args.pretrained_model is None: 


    #modelmerda = SymmetricSiameseModel(model, CLIPTextEncoder(256), args)
    checkpoint = torch.load(args.pretrained_model)
    model.load_state_dict(checkpoint)
    #model = modelmerda.image_encoder



model.eval().cuda()
dataset = ZippedDataloader(args.file,)
projector = TSNEProjector(dataset, model, imsize = args.imsize, mapsize = 20000)

print(
    "Using:\n",
    f"\tProjector: {projector}",
    f"\n\tModel: {model}"
    f"\n\tData: {args.file}"
)
image = projector.place_images()
cv2.imwrite('tmp.png', image)
