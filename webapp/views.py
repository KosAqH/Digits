from flask import Flask, render_template, redirect, session, url_for, request
from . import app

@app.route("/")
def index():
    return render_template("drawboard.html")

def process_image():
    pass

