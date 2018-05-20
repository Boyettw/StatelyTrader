import numpy
import sys
import numpy as np
import pymongo
from pymongo import MongoClient
from multiprocessing import Process
import os

def generate_label(db, collections, input_epoch, label_offset):   #TODO reformat to return the one label based on algorithmic search through each market, return the index in the sorted collection
    market_index = 0
    max_gain = 0.9975
    future_trade_price = 0
    future_epoch = input_epoch + label_offset
    max_market = 35
    fee = (0.9975*0.9975)
    for market_name in collections:
        for label_document in db[market_name].find({'epoch': {'$gte': future_epoch}}).sort([('epoch', pymongo.ASCENDING)]).limit(1):
            future_trade_price = label_document['rate'] * fee
        for trade in db[market_name].find({'epoch': {'$lte': input_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(1):
            current_trade_price = trade['rate'] / fee
            trade_gain = future_trade_price / current_trade_price
            if trade_gain > max_gain:
                max_gain = trade_gain
                max_market = market_index
        market_index += 1
    return max_market


def generate_feature(input_epoch, db, collections, trade_length, history_length, markets_length, ttl_length):
    history = numpy.zeros((ttl_length, markets_length))
    market_count = 0
    input_min = sys.float_info.max
    for market in collections:
        val_index = ttl_length
        for val in db[market].find({'epoch': {'$lte': input_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(history_length):
            if val['epoch'] < input_min:
                input_min = val['epoch']
            history[val_index - 4][market_count] = val['epoch']
            history[val_index - 3][market_count] = val['orderType']
            history[val_index - 2][market_count] = val['rate']
            history[val_index - 1][market_count] = val['quantity']
            val_index -= trade_length
        market_count += 1
    epoch_index = 0
    for market_index in range(0, markets_length):
        history[epoch_index][market_index] -= input_min
    epoch_index += 4

    return history

def find_test_max(db, collections, offset):
    max_epoch = sys.float_info.max
    temp_max_epoch = sys.float_info.max
    for market in collections:
        for trade in db[market].find().sort([('epoch', pymongo.DESCENDING)]).limit(1):
            if temp_max_epoch > trade['epoch']:
                temp_max_epoch = trade['epoch']
    for market in collections:
        for trade in db[market].find({'epoch': {'$lte':temp_max_epoch - (offset + 1)}}).sort([('epoch', pymongo.DESCENDING)]).limit(1):
            if max_epoch > trade['epoch']:
                max_epoch = trade['epoch']
    return max_epoch

def find_test_min(db, collections, test_max):
    min_epoch = sys.float_info.max
    for market in collections:
        for trade in db[market].find({'epoch': {'$lte': test_max}}).sort([('epoch', pymongo.DESCENDING)]).limit(30000):
            if min_epoch > trade['epoch']:
                min_epoch = trade['epoch']
    return min_epoch

def find_train_max(db, collections, test_min): #generate feature hands you
    max_epoch = sys.float_info.max
    for market in collections:
        for trade in db[market].find({'epoch': {'$lte': test_min}}).sort([('epoch', pymongo.DESCENDING)]).limit(102):
            if max_epoch > trade['epoch']:
                max_epoch = trade['epoch']
    return max_epoch    #returns the min - extra trades -

def find_train_min(db, collections, history_length):
    min_epoch = sys.float_info.min # 101 off each market, latest epoch available
    for market in collections:
        for trade in db[market].find().sort([('epoch', pymongo.ASCENDING)]).limit(history_length):
            if trade['epoch'] > min_epoch:
                min_epoch = trade['epoch']
    return min_epoch

def find_ranges(history_length, offset):
    client = MongoClient('127.0.0.1', 27017)
    db = client['markets']
    collections = db.collection_names()
    collections.sort()
    return_dict = {}
    return_dict['test_max'] = find_test_max(db, collections, offset)
    return_dict['test_min'] = find_test_min(db, collections,return_dict['test_max'])
    return_dict['train_max'] = find_train_max(db, collections, return_dict['test_min'] - (offset + 1))
    return_dict['train_min'] = find_train_min(db, collections, history_length)
    client.close()
    return return_dict

def validate_range(db, collections, input_epoch, history_length, label_offset):
    for market in collections:
        last_epoch = 0
        for val in db[market].find({'epoch': {'$lte': input_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(history_length):
            if last_epoch != 0 and (val['epoch'] -  last_epoch) > 120000:
                return False
            last_epoch = val['epoch']
        future_epoch = input_epoch + label_offset
        for val in db[market].find({'epoch': {'$gte': future_epoch}}).sort([('epoch', pymongo.ASCENDING)]).limit(1):
            if val['epoch'] > future_epoch + 240000:
                return False
    return True

def generate_data(batch_size, trade_length, history_length, markets_length, ttl_length, label_offset, num_classes, test_min, train_max, test_max, train_min):   #TODO call search functions and related spaghetti correctly, fix the OO if need be
    print("generating data")
    client = MongoClient('127.0.0.1', 27017)
    db = client['markets']
    collections = db.collection_names()
    collections.sort()
    test_slices = int(int((test_max - test_min) - 1) / int(label_offset))

    return_dict = {}
    return_dict['train_features'] = numpy.zeros((batch_size, ttl_length, markets_length))
    return_dict['test_features'] = numpy.zeros((test_slices, ttl_length, markets_length))
    return_dict['train_labels'] = numpy.zeros(batch_size)
    return_dict['test_labels'] = numpy.zeros(test_slices)
    return_dict['test_epochs'] = numpy.zeros(test_slices)
    accept_count = 1.0
    reject_count = 1.0
    for i in range(0, batch_size):
        train_epoch = np.random.uniform(low=train_min, high=train_max)
        while not validate_range(db, collections, train_epoch, history_length, label_offset):
            reject_count += 1.0
            train_epoch = np.random.uniform(low=train_min, high=train_max)
        accept_count += 1.0
        print(accept_count/reject_count)
        return_dict['train_features'][i] = generate_feature(train_epoch, db, collections,  trade_length, history_length, markets_length, ttl_length)
        return_dict['train_labels'][i] = generate_label(db, collections, train_epoch, label_offset)
#        print("generated batch index: %d" % i)

    print(test_slices)
    for i in range(test_slices):
        test_epoch = (test_min + i * label_offset)
        return_dict['test_epochs'][i] = test_epoch
        return_dict['test_features'][i] = generate_feature(test_epoch, db, collections,  trade_length, history_length, markets_length, ttl_length)
        return_dict['test_labels'][i] = generate_label(db, collections, test_epoch, label_offset)
#        print("generated test_batch index: %d" % i)
#    print(return_dict['train_labels'])
#    print(return_dict['test_labels'])
    directory = 'E:\\'
    np.save(directory + str(label_offset) + '_' + str(history_length) + '_' + 'train_features', return_dict['train_features'])
    np.save(directory + str(label_offset) + '_' + str(history_length) + '_' + 'test_features', return_dict['test_features'])
    np.save(directory + str(label_offset) + '_' + str(history_length) + '_' + 'train_labels', return_dict['train_labels'])
    np.save(directory + str(label_offset) + '_' + str(history_length) + '_' + 'test_labels', return_dict['test_labels'])
    np.save(directory + str(label_offset) + '_' + str(history_length) + '_' + 'test_epochs', return_dict['test_epochs'])
    client.close()
    print('Generated ' + str(label_offset) + '_' + str(history_length))

#def generate_data(batch_size, trade_length, history_length, markets_length, ttl_length, label_offset, num_classes, test_min, train_max, test_max, train_min):   #TODO call search functions and related spaghetti correctly, fix the OO if need be

processes = []
print("started")
if __name__ == '__main__':
    for label_offset in [24120000, 12000000]:
        for history_length in range(25, 100,1):
            ranges = find_ranges(history_length, offset=label_offset)
            processes.append(Process(target=generate_data,args=(50000, 4, history_length, 35, history_length*4, label_offset, 36, ranges['test_min'], ranges['train_max'], ranges['test_max'], ranges['train_min'])))
    print("finished generating processes")
    num_processes = 6
    index = 0

    for i in range(int(len(processes) / num_processes)):
        for j in range(num_processes):
            processes[i * num_processes + j].start()
        for j in range(num_processes):
            processes[i * num_processes + j].join()
            processes[i * num_processes + j].terminate()

    for i in range(int(len(processes) % num_processes)):
        processes[len(processes) - 1 - i].start()
    for i in range(int(len(processes) % num_processes)):
        processes[len(processes) - 1 - i].join()
        processes[len(processes) - 1 - i].terminate()