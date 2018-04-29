import requests
import json
import pymongo
from pymongo import MongoClient
import time
import numpy
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from numpy import genfromtxt, newaxis
from random import randint

def generate_labels(db, collections, input_epoch):
    markets_length = 190
    label_offset = 6 * 1000 * 60
    labels = numpy.zeros(markets_length)
    market_index = 0
    for market_name in collections:
        for label_document in db[market_name].find({'epoch': {'$gte': (input_epoch + label_offset)}}).sort([('epoch', pymongo.ASCENDING)]).limit(1):
            labels[market_index] = label_document['rate']
    return labels


def generate_feature(db, collections, input_epoch, minibatch_size):
    trade_length = 4
    history_length = 100
    markets_length = 190
    ttl_length = markets_length * history_length * trade_length
    history = numpy.zeros(minibatch_size, markets_length, history_length, trade_length)
    market_count = 0
    for market in collections:
        val_index = history_length - 1
        for val in db[market].find({'epoch': {'$lte': input_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(history_length):
            history[market_count][val_index][0] = val['epoch']
            history[market_count][val_index][1] = val['orderType']
            history[market_count][val_index][2] = val['rate']
            history[market_count][val_index][3] = val['quantity']
            val_index -= 1
        market_count = market_count + 1
    return history

def find_test_max():
    max_epoch = sys.float_info.max
    for market in collections:
        for trade in db[market].find().sort([('epoch', pymongo.DESCENDING)]).limit(2):
            if max_epoch > trade['epoch']:
                max_epoch = trade['epoch']
    return max_epoch

def find_test_min(db, collections):
    min_epoch = sys.float_info.min
    for market in collections:
        for trade in db[market].find().sort([('epoch', pymongo.DESCENDING)]).limit(404): #magic 100 + 1 label 4 times
            if trade['epoch'] > min_epoch:
                min_epoch = trade['epoch']
    return min_epoch


def find_train_max(db, collections, test_min):
    max_epoch = sys.float_info.max
    for market in collections:
        for trade in db[market].find({'epoch': {'$lte': test_min}}).sort([('epoch', pymongo.DESCENDING)]).limit(2):
            if max_epoch > trade['epoch']:
                max_epoch = trade['epoch']
    return max_epoch

def find_train_min(db, collections):
    min_epoch = sys.float_info.min # 101 off each market, latest epoch available



def find_train_max(db, collections, test_max):


def generate_features(db, collections):   #TODO call search functions and related spaghetti correctly, fix the OO if need be
    train_max = find_train_max(db, collections)
    test_min = find_test_min(db, collections)
    test_max = find_test_max(db, collections)
    train_min = find_train_min


    return_dict = {}
    return_dict['train_features'] =                     #find_max(db, collections)
    return_dict['train_labels'] =                            #find_dev_min(db, collections)
    return_dict['dev_features'] =                            #find_dev_max(db, collections)
    return_dict['dev_labels'] =                          #fin



def train(db, collections, train_features, train_labels):
    init_range = 1
    num_classes = 190
    minibatch_size = 100
    epochs = 1000
    learnrate = .0003
    num_hidden_units = 100
    num_minibatches = train_features.shape[0] / minibatch_size
    num_steps = train_features.shape[1]
    num_features = num_steps    # hidden layer num of features
    num_inputs = train_features.shape[2]

    x_type = tf.float32
    y_type = tf.int64

    x = tf.placeholder(x_type, [None, num_steps, num_inputs])
    y = tf.placeholder(y_type)

    hidden_func = tf.tanh
    weights = tf.Variable(tf.random_normal([num_hidden_units, num_classes]))
    biases = tf.Variable(tf.random_normal([num_classes]))

    z = tf.unstack(x, num_steps, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden_units, forget_bias=1.0)
    outputs, _ = rnn.static_rnn(lstm_fw_cell, z, dtype=tf.float32)

    logit = tf.matmul(outputs[-1], weights) + biases

    loss_calc = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learnrate).minimize(loss_calc)

    sess = tf.Session()
    # correct_pred = tf.equal(tf.argmax(logit,1), y)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logit, 1), y)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./", sess.graph)
    init = tf.global_variables_initializer()

    sess.run(init)

    update = 0

    for i in range(epochs):
        for j in range(int(train_features.shape[0] / minibatch_size)):
            minibatch_features = train_features[j * minibatch_size:(j + 1) * minibatch_size]
            minibatch_labels = train_labels[j * minibatch_size:(j + 1) * minibatch_size]
            _, loss, acc = sess.run([optimizer, loss_calc, accuracy], feed_dict={x: minibatch_features, y: minibatch_labels})
            update = update + 1
        # test_loss, test_acc = sess.run([loss_calc, accuracy], feed_dict={x: dev_features, y: dev_labels})
        print("Epoch %03d: train=%.3f" % (i, acc))



client = MongoClient('10.0.0.88', 27017)
db = client['markets']
collections = db.collection_names()
collections.sort()
train(db, collections, train_features = np.random.ranf((1000, 400, 190)), train_labels = np.random.randint(low=0, high=190, size=(1000)))
client.close()

"""
train rnn on input = (minibatch_size x 400 x 190)(save to disk after for rententional training?), 
feed flattened? input + output + weights into convolutional with larger fields,
take output classification on all data where rnn was true/positives excluding false positives,
calling this approach grafting, like a skin graft of information to rebuild ability to follow rnn weights(less than 5% of train data),
train on real results for 60%,
for the next 10% predict in section called dev,
then algorithmically determine best fit by listening to all three magnitudes of time,
buying on 30 seconds but selling on 6 minutes to start(compare how often they argue and look at the real results yourself to determine who is more accurate.),
need to set number of classes to 191 (one extra for when leaving coin in btc is optimal,
"""