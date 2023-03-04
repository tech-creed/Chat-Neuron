import os
import sys
from json import load
from flask import Flask

import pymongo

STATIC_FOLDER = sys.path[0] + '/static/'
TEMPLATES_FOLDER = sys.path[0] + '/templates/'
UPLOAD_FOLDER = sys.path[0] + '/static/uploads/'

app = Flask(__name__, template_folder=TEMPLATES_FOLDER, static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MONGO_URI'] = 'mongodb+srv://chat:chat-neuron@cluster0.bnanchq.mongodb.net/chat?retryWrites=true&w=majority'
app.secret_key = "SSKEY"

client = pymongo.MongoClient("mongodb+srv://chat:chat-neuron@cluster0.bnanchq.mongodb.net/chat?retryWrites=true&w=majority")
db = client.db