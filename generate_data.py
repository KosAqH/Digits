## Script for generate more data through some transformation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # only for testing

import os
from datetime import datetime

from img_manipulation import *

def load_data(filename: str = "Data\\train.csv"):
    df = pd.read_csv(filename)
    X = df.iloc[:, 1:].copy().values
    y = df['label'].copy().values
    return (X, y)

def generate_filename(prefix: str = "train"):
    s = f"{prefix} {datetime.today().strftime(r'%Y-%m-%d %H-%M-%S')}.csv"
    return s

def save_data(X, y):
    filename = generate_filename()
    columns = []
    for i in range(len(X[0].flatten())):
        columns.append(f'pixel{i}')
    
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "label", y)

    df.to_csv(filename, index=False)

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
        raise Exception("Wrong type of transformation")

    return new_img

def modify_img(img, transformations: dict):
    m_img = img

    for k in transformations.keys():
        m_img = run_transformation(m_img, k, transformations[k])

    return m_img


def process_image(img):
    modified_imgs = []
    modified_imgs.append(modify_img(img, {"rotate" : 90, "shift" : (3, -3), "noise" : 25}).flatten())

    return modified_imgs

if __name__ == "__main__":
    w = 28
    h = 28

    ## Load all data
    X, y = load_data()

    new_X = []
    new_y = []

    for i, label in enumerate(y):
        img = X[i].reshape(w, h)
        modified_imgs = process_image(img)
        l = len(modified_imgs)
        new_X += modified_imgs
        new_y += l * [label]
        
    ## Save data
    save_data(new_X, new_y)
    

