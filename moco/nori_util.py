import nori2 as nori
import cv2 as cv
import numpy as np
from PIL import Image
import json

fetcher = nori.Fetcher()

def get_img(nori_id):
    img = fetcher.get(nori_id)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv.imdecode(img, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img).convert('RGB')
    
    return img

def get_sample_list_from_json(json_path):
    j = json.load(open(json_path, 'r'))
    samples = []
    for info in j['info_dicts']:
        samples.append((info['nori_id'], info['label']))
    return samples
