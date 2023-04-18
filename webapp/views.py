from flask import Flask, render_template, redirect, session, url_for, request
from . import app, knn, svc
from PIL import Image
from io import BytesIO
import base64
import numpy as np


@app.route("/")
def index():
    return render_template("drawboard.html")

@app.route("/send_img",  methods = ['POST'])
def send_img():
    if request:
        base64_img = request.get_data()
        print(type(base64_img))
        img = process_image(base64_img.decode("ascii") )
        predictment = predict_image(img)

        return predictment
    return "NOT OK"

def process_image(encoded_img):
    starter = encoded_img.find(',')
    image_data = encoded_img[starter+1:]
    image_data = bytes(image_data, encoding="ascii")
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.resize((28,28), Image.Resampling.LANCZOS)
    img = img.convert('LA')
    arr = np.array(img)
    data = arr[:,:,1]
    # img = Image.fromarray(data)
    # img.save('image.png')
    return data

def predict_image(img: np.ndarray):
    knn_result = knn.predict([img.flatten()])[0]
    svc_result = svc.predict([img.flatten()])[0]

    return {"KNN" : int(knn_result), "SVC" : int(svc_result)}