import pymongo
from pymongo import MongoClient
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

def evaluate_labels(db, collections, buy_epoch, sell_epoch):
    market_index = 0
    future_trade_price = 0
    current_trade_price = 0
    gains = []
    for market_name in collections:
        for trade in db[market_name].find({'epoch': {'$lte': buy_epoch}}).sort([('epoch', pymongo.DESCENDING)]).limit(1):
            current_trade_price = trade['rate']
        for label_document in db[market_name].find({'epoch': {'$gte': sell_epoch}}).sort([('epoch', pymongo.ASCENDING)]).limit(1):
            future_trade_price = label_document['rate']

        gains.append(future_trade_price/current_trade_price)
        market_index += 1

    gains.append(1.0)
    num_better = 0.0
    markets_length = market_index
    return gains

def vote(num_votes, labels, last_label):
    label_count = {}
    index = len(labels) - 1
    max_vote = int(num_votes * .9)
    max_label = last_label
    for i in range(num_votes):
        if labels[index - i] in label_count:
            label_count[labels[index - i]] += 1
        else:
            label_count[labels[index - i]] = 1
    for label in label_count:
        if max_vote < label_count[label]:
            max_label = label
            max_vote = label_count[label]
    return max_label



def train(train_features, train_labels, test_features, test_labels, test_epochs, minibatch_size, epochs, learn_rate, num_hidden_units, offset, gpu, num_votes):
    client = MongoClient('127.0.0.1', 27017)
    db = client['markets']
    collections = db.collection_names()
    collections.sort()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True  # allocate dynamically
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
    lstm_fw_cell = rnn.LSTMCell(num_hidden_units, forget_bias=1.0)
    outputs, _ = rnn.static_rnn(lstm_fw_cell, z, dtype=x_type)

    logit = tf.matmul(outputs[-1], weights) + biases

    loss_calc = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss_calc)

    sess = tf.Session()
    """
    For offset in offset set:
        
    """

    with tf.name_scope('accuracy'):
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logit, 1)
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logit, 1), y)
#        with tf.name_scope('performance'):

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
    investment = 0.9975
    total_added = 1.0
    last_label = 35
    trade_count = 0
    avg_gain = 0.0
    buy_epoch = sys.float_info.max
    last_epoch_index = num_steps - 4
    print(last_epoch_index)


    labels = []
    softmaxes = []
    cutoff = num_classes/2
    k = 0
    while k < len(test_labels):
        if investment < 0.1:
            investment  += 1.0
            total_added += 1.0
        current_epoch = test_epochs[k]
        sell_epoch = current_epoch + offset
        if  current_epoch < buy_epoch:
            buy_epoch = current_epoch
        test_loss, test_acc, label, test_logit = sess.run([loss_calc, accuracy, prediction, logit], feed_dict={x: np.expand_dims(test_features[k], axis=0), y: np.expand_dims(test_labels[k], axis=0)})
        labels.append(int(label))
        num_better = 0
        markets_avg_gains = 0.0
        guess = float(test_logit[0][last_label])
        #print(guess)

        for softmax_index in range(len(test_logit[0])):
            #print(test_logit[0][softmax_index])
            if softmax_index != int(last_label) and float(test_logit[0][softmax_index]) > guess:
                num_better += 1
        #print(num_better)
        #label = np.random.randint(0, 36)
        #this should be majority vote(labels over past 10 minutes of epoch agree on swing and return true.
        # if last_label != label:
        if ((len(labels) >= num_votes) and (vote(num_votes, labels, last_label) != last_label)) and (num_better > 7):
            gains = evaluate_labels(db, collections, buy_epoch, sell_epoch)  #max_epoch, label_offset)
            label_gain = gains[int(label)]
            num_better = 0.0
            ttl_percentile = 0.0
            num_labels = len(gains)
            ttl_gains = 0.0
            for l in range(num_labels):
                num_better += 1.0 if (gains[l] > label_gain and label != l) else 0.0
                ttl_gains += gains[l] if l != 35 else 0.0
            local_avg_gain = ttl_gains/(num_labels - 1)
            percentile = (num_labels - (num_better + 1)) / (num_labels - 1)
            alt_train = int(label) if (percentile >= .7 and gains[int(label)] >= 1.0) else test_labels[k]
            _, test_loss, test_acc, test_label = sess.run([optimizer, loss_calc, accuracy, prediction], feed_dict={x: np.expand_dims(test_features[k], axis=0), y: np.expand_dims(alt_train, axis=0)})  # here we can swap the label to the prediction if it evaluates above 80th percentile?

            while k < len(test_labels) and sell_epoch > (test_epochs[k] + 3000):
                k += 1
            print("buy_epoch=%f sell_epoch=%f" % (buy_epoch, sell_epoch))
            buy_epoch = sell_epoch
            investment = (((investment * 0.9975 if last_label != 35 else 1.0) * label_gain) * 0.9975 if label != 35 else 1.0)
            avg_gain += label_gain
            markets_avg_gains += local_avg_gain
            trade_count += 1
            percentiles += percentile
            print("percentile=%.3f, gain=%.3f, investment=%.3f, last_label=%s, label=%s" % (percentile, label_gain, investment, last_label, label))
            last_label = label
            labels = []
        k += 1
        #_, train_test_loss, test_acc, label = sess.run([optimizer, loss_calc, accuracy, prediction], feed_dict={x: np.expand_dims(test_features[k], axis=0), y: np.expand_dims(test_labels[k], axis=0)})

#    if last_label != 35:
#        percentile, gain = evaluate_label(db, collections, int(label), buy_epoch, sell_epoch)
#        print("buy_epoch=%.3f sell_epoch=%.3f" % (buy_epoch, sell_epoch))
#        investment = gain
#        avg_gain += gain
#        trade_count += 1
#        percentiles += percentile
#        print("percentile=%.3f, gain=%.3f, investment=%.3f, last_label=%s, label=%s" % (percentile, gain, investment, last_label, 35))
    if trade_count > 0:
        print("avgPercentile=%.3f, totalDollarsEarned=%.3f, avgGain=%.3f, av_gain_across_markets=%.3f, trade_count=%d" % (percentiles / trade_count, investment - total_added, avg_gain/trade_count, markets_avg_gains/trade_count, trade_count))
    else:
        print("Markets_too_lossy, no trades made.")
    client.close()
offset = 250000
history_length = 9
gpu = 3
num_votes = 75
train(np.load("E:\\" + str(offset) + "_" + str(history_length) + "_train_features.npy"),
      np.load("E:\\" + str(offset) + "_" + str(history_length) + "_train_labels.npy"),
      np.load("E:\\" + str(offset) + "_" + str(history_length) + "_test_features.npy"),
      np.load("E:\\" + str(offset) + "_" + str(history_length) + "_test_labels.npy"),
      np.load("E:\\" + str(offset) + "_" + str(history_length) + "_test_epochs.npy"),
      1000, 500, .0001, 50, offset, gpu, num_votes)


