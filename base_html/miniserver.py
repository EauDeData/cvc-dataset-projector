from typing import *
from flask import Flask, request, Response, redirect, send_from_directory, render_template, send_file
import requests
import os
import json
import warnings
import clip 
import annoy 
import dill 
import torch
import urllib.parse

warnings.warn("Example server. Security non guaranteed in any case!")

from pytools.models import CLIPLoader

image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
api = Flask(__name__, root_path = "web/", template_folder = "./")
clipmodel = CLIPLoader()

config = json.load(open('config.json', 'r'))
dataset = dill.load(open(config['dataset'], 'rb'))
annoyer = annoy.AnnoyIndex(512, 'angular')
annoyer.load(config['annoy'])

@api.route("/<whatever>.css")
def css(whatever):
    return send_file(whatever + '.css')

@api.route("/<whatever>.js")
def js(whatever):
    return send_file(whatever + '.js')

@api.route("/tiles/<path:numbers>/<file>.png")
def tiles(numbers, file):
    return send_file(os.path.join("tiles", numbers, file + ".png"))

@api.route('/query/<query>')
def get_images_path_by_text_query(query):
    
    with torch.no_grad():
        encoded_query = clipmodel.model.encode_text(clip.tokenize([query]).cuda())[0].cpu().numpy()
        idxs = annoyer.get_nns_by_vector(encoded_query, n = 10)
        files = [dataset[idx] for idx in idxs]
        jsoned = json.dumps({
            "images": files
        })
    print(jsoned)
    return jsoned

@api.route("/")
def index():
    return render_template("index.html")

@api.route("/map.html")
def map_index():
    return render_template("map.html")

@api.route('/<path:path>/<filename>',)
def get_image(path, filename):
    return send_file(urllib.parse.quote(os.path.join(path, filename)))

api.run(host=config["IP"], port=int(config["PORT"]))