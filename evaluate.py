import pymongo
from pymongo import MongoClient
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

def evaluate_label(current_label, last_label, buy_epoch, sell_epoch):
    client = MongoClient('127.0.0.1', 27017)
    db = client['markets']
    collections = db.collection_names()
    collections.sort()
    market_index = 0
    future_trade_price = 0
    current_trade_price = 0
    gains = []
    for market_name in collections:
        if market_index == 35 or market_index == last_label:
            purchase_fee = 1
        else:
            purchase_fee = 0.9975
        for trade in db[market_name].find({'epoch': {'$lte': buy_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(1):
            current_trade_price = trade['rate'] / purchase_fee

        if current_label != 35:
            sell_fee = 0.9975
        else:
            sell_fee = 1

        for label_document in db[market_name].find({'epoch': {'$gte': sell_epoch}}).sort([('epoch', pymongo.ASCENDING)]).limit(1):
            future_trade_price = label_document['rate'] * sell_fee

        gains.append(future_trade_price/current_trade_price)
        market_index += 1

    gains.append(1.0 * sell_fee)
    num_better = 0.0
    markets_length = market_index
    for i in range(len(gains)):
        if gains[i] > gains[current_label]:
            num_better += 1.0
    client.close()
    return (1.0 - (num_better / markets_length)), gains[current_label]

def train(train_features, train_labels, test_features, test_labels, test_epochs, minibatch_size, epochs, learn_rate, num_hidden_units, offset):
    num_steps = train_features.shape[1]
    num_inputs = train_features.shape[2]
    num_classes = num_inputs + 1
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
    sess.run(init)


    print(train_labels)
    print(test_labels)
    for i in range(epochs):
        for j in range(int(train_features.shape[0] / minibatch_size)):
            minibatch_features = train_features[j * minibatch_size:(j + 1) * minibatch_size]
            minibatch_labels = train_labels[j * minibatch_size:(j + 1) * minibatch_size]
            _, loss, acc = sess.run([optimizer, loss_calc, accuracy], feed_dict={x: minibatch_features, y: minibatch_labels})
        test_loss, test_acc = sess.run([loss_calc, accuracy], feed_dict={x: test_features, y: test_labels})
        print("Epoch %03d: train=%.3f test=%.3f" % (i, acc, test_acc))

    percentiles = 0.0
    investment = 1.0
    last_label = 35
    trade_count = 1
    avg_gain = 0.0
    buy_epoch = sys.float_info.max
    last_epoch_index = num_steps - 4
    print(last_epoch_index)

    for k in range(len(test_labels)):
        current_epoch = test_epochs[k]
        sell_epoch = current_epoch + offset
        if  current_epoch < buy_epoch:
            buy_epoch = current_epoch
        test_loss, test_acc, label = sess.run([loss_calc, accuracy, prediction], feed_dict={x: np.expand_dims(test_features[k], axis=0), y: np.expand_dims(test_labels[k], axis=0)})
        #label = np.random.randint(0, 36)

        if label != last_label:
            percentile, gain = evaluate_label(int(label), last_label, buy_epoch, sell_epoch)  #max_epoch, label_offset)
            buy_epoch = sys.float_info.max
            investment *= gain
            avg_gain += gain
            trade_count += 1
            percentiles += percentile
            print("percentile=%.3f, gain=%.3f, investment=%.3f, last_label=%s, label=%s" % (percentile, gain, investment, last_label, label))
        last_label = label
    print("avgPercentile=%.3f, totalGain=%.3f, avgGain=%.3f, trade_count=%d" % (percentiles / trade_count, investment, avg_gain/trade_count, trade_count))
train(np.load("E:\\120000_1_train_features.npy"),
      np.load('E:\\120000_1_train_labels.npy'),
      np.load('E:\\120000_1_test_features.npy'),
      np.load('E:\\120000_1_test_labels.npy'),
      np.load('E:\\120000_1_test_epochs.npy'),
      20000,
      100,
      .1,
      4,
      120000)


