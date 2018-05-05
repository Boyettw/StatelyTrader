import pymongo
from pymongo import MongoClient
import time

client = MongoClient('127.0.0.1', 27017)
db = client['markets']
collections = db.collection_names()
collections.sort()
count = 0
market_count = 0
for market in collections:
    tempcount = 0
    print(market)
    market_count += 1
    for val in db[market].find():
        tempcount += 1
    print(tempcount)
print(count)
print(market_count)
client.close()

