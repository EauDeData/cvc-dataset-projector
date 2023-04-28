import torch
import torchvision
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import cv2

# https://open.spotify.com/track/5NSR9uH0YeDo67gMiEv13n?si=1885da5af0694dfd
from dataclasses import dataclass
from typing import *

@dataclass
class TripletOutput:
    loss: torch.Tensor
    anchor_embedding: torch.Tensor
    positive_embedding: torch.Tensor
    negative_embedding: Optional[torch.Tensor] = None


class Resnet(torch.nn.Module):
    def __init__(self, norm = 2, resnet = '152', pretrained = 'imagenet'):
        super(Resnet, self).__init__()

        if resnet == '152': self.resnet = torchvision.models.resnet152(pretrained = pretrained)
        elif resnet == '101': self.resnet =  torchvision.models.resnet101(pretrained = pretrained)
        elif resnet == '50': self.resnet =  torchvision.models.resnet50(pretrained = pretrained)
        elif resnet == '34': self.resnet =  torchvision.models.resnet34(pretrained = pretrained)
        elif resnet == '18': self.resnet =  torchvision.models.resnet18(pretrained = pretrained)
        else: raise NotImplementedError
        
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.norm = norm

    def infer(self, image):
        # Single Numpy Array inference
        with torch.no_grad():

            return self(torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)).cpu().squeeze().view(-1).numpy()
        

    def __str__(self):
        return str(self.resnet)


    def forward(self, batch):

        h = self.resnet(batch)
        if self.norm is not None: h =  torch.nn.functional.normalize(h, p = self.norm, dim = 1)
        return h



class ResNetWithEmbedder(torch.nn.Module):
    def __init__(self, resnet='152', pretrained=True, embed_size: int = 512):
        super(ResNetWithEmbedder, self).__init__()

        if resnet == '152':
            resnet = torchvision.models.resnet152(pretrained=pretrained)
        elif resnet == '101':
            resnet = torchvision.models.resnet101(pretrained=pretrained)
        elif resnet == '50':
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif resnet == '34':
            resnet = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet == '18':
            resnet = torchvision.models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError
    
        self.trunk = resnet
        trunk_output_size = self.trunk.fc.in_features
        self.trunk.fc = torch.nn.Identity()
        self.embedder = torch.nn.Linear(trunk_output_size, embed_size)
        
    def infer(self, image):
        # Single Numpy Array inference
        image = image / 255
        image = image - np.array([0.4850, 0.4560, 0.4060]) 
        image = image / np.array([0.2290, 0.2240, 0.2250])
        with torch.no_grad():
            return self(torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).cuda()).cpu().squeeze().view(-1).numpy()

    def __str__(self):
        return str(self.trunk)

    def forward(self, batch):
        h = self.trunk(batch)
        h = self.embedder(h)
        return h
    
class CLIPLoader(torch.nn.Module):
    name = 'CLIP_mapper'
    def __init__(self, device = 'cuda', *args, **kwargs) -> None:
        super(CLIPLoader, self).__init__()
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit = False)
        self.model.to(device)

    def _predict(self, text):
        tokens = clip.tokenize(text).to(self.device)
        return self.model.encode_text(tokens)

    def _encode_image(self, img):
        image = Image.fromarray(img)
        img = self.preprocess(image).to(self.device).unsqueeze(0)

        with torch.no_grad():
            
            return self.model.encode_image(img)
    
    def infer(self, img):
        return self._encode_image(img).unsqueeze(0).cpu().squeeze().view(-1).numpy()

import torch

from typing import List
from transformers import CLIPTextModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
    

class CLIPTextEncoder(torch.nn.Module):
    """
    Wrapper around CLIPTextModel. Adds a projection layer to the output of the CLIPTextModel.
    """
    def __init__(self, embed_size: int = 256, freeze_backbone: bool = True):
        super(CLIPTextEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        if freeze_backbone:
            for param in self.text_model.parameters():
                param.requires_grad = False

        self.projector = torch.nn.Linear(self.text_model.config.hidden_size, embed_size)

    def tokenizer_encode_text(self, text: List[str]) -> BatchEncoding:
        """
        Use the tokenizer to encode the text.

        Args:
            text: str
        Returns:
            torch.Tensor of shape (batch_size, max_seq_len)
        """
        tokenized = self.tokenizer(text, return_tensors="pt", padding=True)
        return tokenized

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: torch.Tensor of shape (batch_size, max_seq_len)
        Returns:
            torch.Tensor of shape (batch_size, embed_size)
        """
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[1]
        return self.projector(text_features)

from torch.nn import functional as F


class SymmetricSiameseModel(torch.nn.Module):
    def __init__(self, image_encoder: torch.nn.Module, text_encoder: torch.nn.Module, args):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.parameter.Parameter(torch.tensor(1.0))
        self.loss_fn = None

    def tokenize(self, text: List[str]) -> BatchEncoding:
        return self.text_encoder.tokenizer_encode_text(text)

    def forward(
            self,
            anchor_image: torch.Tensor, 
            text_input_ids: torch.Tensor,
            text_attention_mask: torch.Tensor,
            ) -> TripletOutput:
        # Compute the embeddings
        image_embedding = self.image_encoder(anchor_image)
        text_embedding = self.text_encoder(text_input_ids, text_attention_mask)
        # L2-normalize the embeddings
        image_embedding_norm = F.normalize(image_embedding, dim=1, p=2)
        text_embedding_norm = F.normalize(text_embedding, dim=1, p=2)
        # Compute the logits
        logits = torch.mm(image_embedding_norm, text_embedding_norm.T) * torch.exp(self.temperature)
        # Compute the symmetric cross entropy loss
        loss = self.loss_fn(logits)
        return TripletOutput(
            loss=loss,
            anchor_embedding=image_embedding,
            positive_embedding=text_embedding,
        )
