## Script for generate more data through some transformation
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # only for testing
import itertools

# import os
import json
from datetime import datetime

from img_manipulation import *

def load_data(filename: str = "Data\\train.csv"):
    df = pd.read_csv(filename)
    X = df.iloc[:, 1:].copy().values
    y = df['label'].copy().values
    return (X, y)

def generate_filename(prefix: str = "train", extension = ".csv"):
    s = f"{prefix} {datetime.today().strftime(r'%Y-%m-%d %H-%M-%S')}{extension}"
    return s

def save_data(X, y):
    filename = generate_filename()
    columns = []
    for i in range(len(X[0].flatten())):
        columns.append(f'pixel{i}')

    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "label", y)

    print(df.info(verbose=False, memory_usage="deep"))

    df.to_csv(filename, index=False)

def save_combinations(c):
    fname = generate_filename(prefix="combinations")
    with open(fname, "w") as f:
        json.dump(c,f)

def run_transformation(img, transformation_type: str = "rotate", val = None):
    if transformation_type == "rotate":
        if type(val) is int:
            new_img = rotate_image(img, val)
        else:
            raise Exception("Wrong val. It should be int.")

    elif transformation_type == "shift":
        if type(val) is tuple and len(val) == 2:
            new_img = shift_image(img, val[0], val[1])
        else:
            raise Exception("Wrong val. It should be tuple with two elements.")
        
    elif transformation_type == "noise":
        if type(val) is int:
            new_img = add_noise(img, val)
        else:
            raise Exception("Wrong val. It should be int.")

    elif transformation_type == "zoom":
        if type(val) is float:
            new_img = zoom_image(img, val)
        else:
            raise Exception("Wrong type of val. It should be float.")
    else:
        raise Exception(f"Wrong type of transformation: {transformation_type}")
    
    return new_img

def modify_img(img, transformations: dict):
    m_img = img

    for k in transformations.keys():
        m_img = run_transformation(m_img, k, transformations[k])

    return m_img.flatten()

def generate_combinations(d):
    keys, values = zip(*d.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts

def process_image(img, combinations):
    modified_imgs = []
    for c in combinations:
        modified_imgs.append(modify_img(img, c))
    return modified_imgs

if __name__ == "__main__":
    w = 28
    h = 28

    c = []
    c += generate_combinations({ "zoom": [0.8, 1.], "rotate": [10, 20, -10, -20], "shift": [(0,4), (4,0), (0,-4), (-4, 0), (4,4), (-4,-4), (-4,4), (4,-4)]})
    c += generate_combinations({ "zoom": [1.25, 1.5], "rotate": [10, -10]})

    ## Load all data
    X, y = load_data()

    new_X = []
    new_y = []

    for i, label in enumerate(y):
        img = X[i].reshape(w, h)
        modified_imgs = process_image(img, c)
        l = len(modified_imgs)
        new_X += modified_imgs
        new_y += l * [label]
        
        if i % 100 == 0:
            print(i)
        
    ## Save data
    save_data(np.array(new_X), np.array(new_y))
    save_combinations(c)
    

