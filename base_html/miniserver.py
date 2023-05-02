from typing import *
from flask import Flask, request, Response, redirect, send_file
import requests
import os
import pickle
import json
import warnings
import clip 
import annoy 

warnings.warn("Example server. Security non guaranteed in any case!")

from pytools.models import CLIPLoader

image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
api = Flask(__name__)
clip = CLIPLoader()

config = json.load(open('config.json', 'r'))
dataset = pickle.load(open(config['dataset'], 'rb'))
annoyer = annoy.AnnoyIndex(512, 'angular')
annoyer.load(config['annoy'])

@api.route('/text-query/<query>',  methods=['POST'])
def get_images_path_by_text_query(query):
    
    encoded_query = clip.tokenize([query])[0].numpy()
    idxs = annoyer.retrieve_by_vector(encoded_query)
    files = [annoyer.datase.files[idx] for idx in idxs]

    return json.dumps({
        "images": files
    })

@api.route(f'/<path>/<filename:re:[\w]+\.({"|".join(image_extensions)})>',)
def get_image(path, filename):
    return send_file(os.path.join(path, filename), mimetype='image/gif')



api.run(host=config["IP"], port=int(config["PORT"]))
