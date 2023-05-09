from flask import Flask, render_template, redirect, session, url_for, request
from . import app, knn, svc, knn_z, svc_z
from PIL import Image
from io import BytesIO
import base64
import numpy as np

from .zoom import *


@app.route("/")
def digits_drawboard():
    return render_template("drawboard.html")

@app.route("/test_digits")
def digits_test():
    return render_template("drawboard_test.html")

@app.route("/send_img",  methods = ['POST'])
def send_img():
    if request:
        base64_img = request.get_data()
        print(type(base64_img))
        img, img_z = process_image(base64_img.decode("ascii") )
        predictment = predict_image(img, img_z)

        return predictment
    return "NOT OK"

LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def process_image(encoded_img):
    starter = encoded_img.find(',')
    image_data = encoded_img[starter+1:]
    image_data = bytes(image_data, encoding="ascii")
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.resize((28,28), Image.Resampling.LANCZOS)
    img = img.convert('LA')
    arr = np.array(img)
    data = arr[:,:,1]

    img_z  = zoom_centre(data)
    # img = Image.fromarray(data)
    # img.save('image.png')
    return (data, img_z)

def predict_image(img: np.ndarray, img_z, is_test: bool = False):
    flat_img = img.flatten()

    knn_result = knn.predict([flat_img])[0]
    svc_result = svc.predict([flat_img])[0]
    knn_z_result = knn_z.predict([img_z.flatten()])[0]
    svc_z_result = svc_z.predict([img_z.flatten()])[0]
    
    d = {"predictment": {
            "KNN" : int(knn_result), 
            "SVC" : int(svc_result), 
            "KNN_Z" : int(knn_z_result), 
            "SVC_Z" : int(svc_z_result)},
        }

    if is_test:
        return d
    
    else:
        knn_proba = knn.predict_proba([img.flatten()])[0]
        knn_proba = dict(zip(LABELS, knn_proba))
        knn_z_proba = knn_z.predict_proba([img_z.flatten()])[0]
        knn_z_proba = dict(zip(LABELS, knn_z_proba))
        d["proba_knn"] = knn_proba
        d["proba_knn_z"] = knn_z_proba
        return d