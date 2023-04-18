from flask import Flask

app = Flask(__name__)
app.secret_key = 'TMP_SECRET_KEY'

from . import views