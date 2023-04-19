## Script for generate more data through some transformation
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    ## Load all data
    X, y = load_data()
    print(X.shape)
    print(y.shape)

    for i, label in enumerate(y):
        img = X[i,:]
        print(img.shape)
        break
    
    ## Save data
    

