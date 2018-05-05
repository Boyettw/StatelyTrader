import pymongo
from pymongo import MongoClient
import numpy
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from multiprocessing import Process, Queue
import requests
client = MongoClient('127.0.0.1', 27017)
client.drop_database('markets')
db = client.get_database('markets')
r = requests.get('https://bittrex.com/api/v1.1/public/getmarkets')
text = r.json()
names = []
for collection in db.collection_names():
    names.append(collection)
for market in text['result']:
    print(market['MarketName'])
    if market['MarketName'][:3] == "BTC":
        db.create_collection(market['MarketName'])
        db[market['MarketName']].create_index([('epoch', pymongo.DESCENDING)])
