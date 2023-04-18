from flask import Flask
import joblib

app = Flask(__name__)
app.secret_key = 'TMP_SECRET_KEY'

knn = joblib.load("knn.joblib")
svc = joblib.load("svc.joblib")


from . import views