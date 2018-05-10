import pymongo
from pymongo import MongoClient
import numpy
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from random import randint

from sortedcontainers import sortedlist
"""
def buy(market_name, quantity, price):


def sell(market_name, quantity, price):

def transaction():
#make connection
#generate prediction
#look up all coins we have
#sell all coins we have
#
"""

"""
Reversed iterator over sorted dict allows you to remove trailing values and nlogn sorted insertion time, logn lookuptime, constant time trimming.

"""
def evaluate_label(label_index, input_epoch, label_offset):
    client = MongoClient('127.0.0.1', 27017)
    db = client['markets']
    collections = db.collection_names()
    collections.sort()

    market_index = 0
    future_trade_price = 0

    future_epoch = input_epoch + label_offset
    markets = []
    gains = []
    for market_name in collections:
        markets.append(market_name)
        for label_document in db[market_name].find({'epoch': {'$lte': future_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(1):
            future_trade_price = label_document['rate']
        for trade in db[market_name].find({'epoch': {'$lte': input_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(1):
            gains.append((future_trade_price - trade['rate']) / trade['rate'])
        market_index += 1
    gains.append(0.0)
    num_better = 0
    markets_length = market_index
    for i in range(len(gains)):
        if gains[i] > gains[label_index]:
            num_better += 1
    client.close()
    return (1 - (num_better / markets_length)), gains[label_index]

def generate_label(db, collections, input_epoch, label_offset, num_classes):   #TODO reformat to return the one label based on algorithmic search through each market, return the index in the sorted collection
    market_index = 0
    max_gain = 0
    future_trade_price = 0
    future_epoch = input_epoch + label_offset
    max_market = num_classes
    for market_name in collections:
        for label_document in db[market_name].find({'epoch': {'$lte': future_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(1):
            future_trade_price = label_document['rate']
        for trade in db[market_name].find({'epoch': {'$lte': input_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(1):
            trade_gain = (future_trade_price - trade['rate']) / trade['rate']
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
    #for market_index in range(0, markets_length):
    #    history[epoch_index][market_index] -= input_min
    epoch_index += 4

    return history

def find_test_max(db, collections):
    max_epoch = sys.float_info.max
    for market in collections:
        for trade in db[market].find().sort([('epoch', pymongo.DESCENDING)]).limit(2):
            if max_epoch > trade['epoch']:
                max_epoch = trade['epoch']
    return max_epoch

def find_test_min(db, collections):
    min_epoch = sys.float_info.max
    for market in collections:
        for trade in db[market].find().sort([('epoch', pymongo.DESCENDING)]).limit(3000):
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

def find_ranges(history_length):
    client = MongoClient('127.0.0.1', 27017)
    db = client['markets']
    collections = db.collection_names()
    collections.sort()
    return_dict = {}
    return_dict['test_min'] = find_test_min(db, collections)
    return_dict['train_max'] = find_train_max(db, collections, return_dict['test_min'])
    return_dict['test_max'] = find_test_max(db, collections)
    return_dict['train_min'] = find_train_min(db, collections, history_length)
    client.close()
    return return_dict

def generate_data(batch_size, trade_length, history_length, markets_length, ttl_length, label_offset, num_classes, test_min, train_max, test_max, train_min):   #TODO call search functions and related spaghetti correctly, fix the OO if need be
    client = MongoClient('127.0.0.1', 27017)
    db = client['markets']
    collections = db.collection_names()
    collections.sort()
    test_batch_size = 300

    return_dict = {}
    return_dict['train_features'] = numpy.zeros((batch_size, ttl_length, markets_length))
    return_dict['test_features'] = numpy.zeros((test_batch_size, ttl_length, markets_length))
    return_dict['train_labels'] = numpy.zeros(batch_size)
    return_dict['test_labels'] = numpy.zeros(test_batch_size)
    for i in range(0, batch_size):
        train_epoch = np.random.uniform(low=train_min, high=train_max)
        return_dict['train_features'][i] = generate_feature(train_epoch, db, collections,  trade_length, history_length, markets_length, ttl_length)
        return_dict['train_labels'][i] = generate_label(db, collections, train_epoch, label_offset, markets_length)
#        print("generated batch index: %d" % i)

    for i in range(0, test_batch_size):
        test_epoch = np.random.uniform(low=test_min, high=test_max)
        return_dict['test_features'][i] = generate_feature(test_epoch, db, collections,  trade_length, history_length, markets_length, ttl_length)
        return_dict['test_labels'][i] = generate_label(db, collections, test_epoch, label_offset, markets_length)
#        print("generated test_batch index: %d" % i)
#    print(return_dict['train_labels'])
#    print(return_dict['test_labels'])
    client.close()
    return return_dict


def train(trade_length, history_length, markets_length, batch_size, minibatch_size, epochs, learn_rate, num_hidden_units, num_inputs, num_classes, label_offset):
    ttl_length = history_length * trade_length
    num_steps = history_length * trade_length

    x_type = tf.float32
    y_type = tf.int64

    x = tf.placeholder(x_type, [None, num_steps, num_inputs])
    y = tf.placeholder(y_type)

    weights = tf.Variable(tf.random_normal([num_hidden_units, num_classes]))
    biases = tf.Variable(tf.random_normal([num_classes]))

    z = tf.unstack(x, num_steps, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden_units, forget_bias=1.0)
    outputs, _ = rnn.static_rnn(lstm_fw_cell, z, dtype=x_type)

    logit = tf.matmul(outputs[-1], weights) + biases

    loss_calc = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss_calc)

    sess = tf.Session()
    with tf.name_scope('accuracy'):
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logit, 1)
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logit, 1), y)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, x_type))
    tf.summary.scalar('accuracy', accuracy)
    init = tf.global_variables_initializer()

    """
    last_process = 0
    queues = []
    processes = []
    num_processes = 1
    for i in range(num_processes):
        queues.append(Queue())
        processes.append(Process(target=generate_data, args=(batch_size, trade_length, history_length, markets_length, ttl_length, queues[i])))
        processes[i].start()
    """
    sess.run(init)
    ranges = find_ranges(history_length)
    for batch in range(1000):
        train_features = 0
        train_labels = 0
        test_features = 0
        test_labels = 0
        """
        last_process = (last_process + 1) % num_processes
        data_dict = queues[last_process].get()
        processes[last_process].join()
        processes[last_process].terminate()
        processes[last_process] = Process(target=generate_data, args=(batch_size, trade_length, history_length, markets_length,ttl_length, queues[last_process]))
        processes[last_process].start()
        """
        data_dict = generate_data(batch_size, trade_length, history_length, markets_length, ttl_length, label_offset, num_classes, ranges['test_min'], ranges['train_max'], ranges['test_max'], ranges['train_min'])
        train_features = data_dict['train_features']
        train_labels = data_dict['train_labels']
        test_features = data_dict['test_features']
        test_labels = data_dict['test_labels']
        print(train_labels)
        print(test_labels)
        for i in range(epochs):
            for j in range(int(train_features.shape[0] / minibatch_size)):
                minibatch_features = train_features[j * minibatch_size:(j + 1) * minibatch_size]
                minibatch_labels = train_labels[j * minibatch_size:(j + 1) * minibatch_size]
                _, loss, acc = sess.run([optimizer, loss_calc, accuracy], feed_dict={x: minibatch_features, y: minibatch_labels})
            test_loss, test_acc = sess.run([loss_calc, accuracy], feed_dict={x: test_features, y: test_labels})
            print("Epoch %03d: train=%.3f test=%.3f" % (i, acc, test_acc))
            if test_acc >= 0.15:
                percentiles = 0.0
                gains = 1.0
                for k in range(len(test_labels)):
                    max_epoch = 0.0
                    for l in range(markets_length):
                        if test_features[k][(history_length - 1) * 4][l] > max_epoch:
                            max_epoch = test_features[k][(history_length - 1) * 4][l]
                    test_loss, test_acc, label = sess.run([loss_calc, accuracy, prediction], feed_dict={x: np.expand_dims(test_features[k], axis=0), y: np.expand_dims(test_labels[k], axis=0)})
                    percentile, gain = evaluate_label(int(label), max_epoch, label_offset)
                    # print(percentile)
                    gains = gains + gains * gain
                    print(gains)
                    percentiles += percentile
                print("percentile=%.3f, averageGain=%.5f" % (
                (percentiles / float(len(test_labels))), gains - 1.0 / float(len(test_labels))))

        percentiles = 0.0
        gains = 1.0
        last_label = 72
        trade_count = 1
        for k in range(len(test_labels)):
            max_epoch = 0.0
            for l in range(markets_length):
                if test_features[k][(history_length - 1)*4][l] > max_epoch:
                    max_epoch = test_features[k][(history_length - 1)*4][l]
            test_loss, test_acc, label = sess.run([loss_calc, accuracy, prediction], feed_dict={x: np.expand_dims(test_features[k], axis=0), y: np.expand_dims(test_labels[k], axis=0)})
            fee = 0.00025 if last_label != label else 0.0
            percentile, gain = evaluate_label(int(label), max_epoch, label_offset)
            #print(percentile)
            gains = gains * (1.0 - fee)
            gains = gains + gains * gain
            trade_count += 1
            last_label = label
            print(gains)
            percentiles += percentile
            #is there a relationship between percentile performance on a "dev" section and total gain on a test? can we learn said relationship?
        print("avgPercentile=%.3f, averageGain=%.5f" % ((percentiles / float(len(test_labels))), (gains - 1.0) / trade_count))

"""
for history_length in range(stop=200, step=5):
    for minibatch_size in range(stop=500, step=100):
        batch_size = 5 * minibatch_size
        for num_hidden_units in range(stop=200, step=5):
            for label_offset in range(start=30000, stop=600000, step = 10000):
"""
train(trade_length=4, history_length=10 , markets_length=72, batch_size=10000, minibatch_size=1000, epochs=300, learn_rate=.003, num_hidden_units=7, num_inputs=72, num_classes=73, label_offset=30000)

"""
train rnn on input = (minibatch_size x 400 x 190)(save to disk after for retentional training?), 
feed flattened? input + output + weights into convolutional with larger fields,
take output classification on all data where rnn was true/positives excluding false positives,
calling this approach grafting, like a skin graft of information to rebuild ability to follow rnn weights(less than 5% of train data),
train on real results for 60%,
for the next 10% predict in section called dev,
then algorithmically determine best fit by listening to all three magnitudes of time,
buying on 30 seconds but selling on 6 minutes to start(compare how often they argue and look at the real results yourself to determine who is more accurate.),
need to set number of classes to 191 (one extra for when leaving coin in btc is optimal,
if optimal search returns a differential of under .25% keep coin in btc, label = int(190),
"""