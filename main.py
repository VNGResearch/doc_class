from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple
import os, glob
import gensim
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import pdb

Document = namedtuple('Document', 'url topic_id doc_no words tags topic')

class RNN(object):

    def step(self):
        pass

class Log(object):
    @staticmethod
    def info(sender, message):
        print('---INFO: ', sender, message, sep='\t')

    @staticmethod
    def warn(sender, message):
        print('---WARN: ', sender, message, sep='\t')

    @staticmethod
    def error(sender, message):
        sys.stderr.write('---ERROR: ', sender, message, '\n', sep='\t')

#return multiple generator
def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen

@multigen
def read_corpus(data_dir, from_percent, to_percent):
    #corpus = []
    topic_id = -1
    doc_id = -1
    doc_count = -1
    for filename in glob.iglob(data_dir + '*.tsv'):
        topic_id += 1
        doc_count = -1
        with open(filename) as f:
            #Log.info('read_corpus', '{}\t{}'.format(topic_id, os.path.basename(filename)))
            #TODO seek to the right part for reading
            docs = f.readlines()
            doc_len = float(len(docs))
            for doc in docs:
                doc_count +=1
                percent = (doc_count+1)/float(doc_len)
                if percent>=to_percent:
                    break
                if percent<from_percent:
                    continue

                doc_id += 1
                parts = doc.split('\t')
                words = ' '.join(part.strip() for part in parts[1:])#concat title, descrpition, content and labels
                if token_type == 'word':
                    words = gensim.utils.to_unicode(words).split()
                if token_type == 'vi_token':
                    raise NotImplementedError()
                #pdb.set_trace()
                #corpus.append(Document(parts[0], topic_id, doc_count, words, [doc_id]))
                yield Document(parts[0], topic_id, doc_count, words, [doc_id], os.path.basename(filename))
    #return corpus

#=======================MAIN

data_dir = '../crawl_news/data/zing/'
train_percent = 0.6
valid_percent = 0.2
test_percent = 0.2
token_type = 'word'
vocabs, word2id, id2word, vocab_size = None, None, None, None

max_seq_len = 200#TODO:change later when uses full document words
lstm_hidden_size = 100
class_size = 60
batch_size = 500

train_full = read_corpus(data_dir, 0, train_percent)
test_full = read_corpus(data_dir, train_percent, 1.0)
train_small = read_corpus(data_dir, 0.2, 0.3)
test_small = read_corpus(data_dir, train_percent, train_percent + 0.05)
train_data = train_small
test_data = test_small

def doc_iter(docs):
    for doc in docs:
        yield ' '.join(doc.words)

def build_vocab():
    global vocabs, word2id, id2word, vocab_size
    countvec = CountVectorizer(min_df=0.01, max_df=0.8)
    countvec.fit(doc_iter(train_data))
    word2id = {word:i for i, word in enumerate(countvec.vocabulary_.keys())}
    id2word = {i:word for i, word in enumerate(countvec.vocabulary_.keys())}

    vocab_size = len(id2word.keys())
    word2id['UNKNOWN'] = vocab_size
    id2word[vocab_size] = 'UNKNOWN'
    vocab_size +=1
    word2id['BLANK'] = vocab_size
    id2word[vocab_size] = 'BLANK'
    vocab_size +=1
    vocabs = set([key for key in word2id.keys()])

    return vocabs, word2id, id2word

def get_input_words_vec(words):
    words = words[0:max_seq_len]
    if len(words)<max_seq_len:
        words.extend(['BLANK']*(max_seq_len-len(words)))

    X = np.zeros(shape=(len(words), vocab_size))
    for i, w in enumerate(words):
        if w not in vocabs:
            w = 'UNKNOWN'
        X[i][word2id[w]] = 1
    return X

def score(sess, predict_op, input_vecs, test_docs):
    total = 0
    correct = 0
    for doc in test_docs:
        vecs = get_input_words_vec(doc.words)
        ret = sess.run([predict_op], feed_dict={input_vecs:vecs, })
        total +=1
        cat_id = np.argmax(ret[0][0])
        if cat_id == doc.topic_id:
            correct +=1
    return correct/float(total) 

def run1():
    build_vocab()

    with tf.Graph().as_default():
        Log.info(None, 'Structure the graph...')
        #define the model
        #input_vecs = tf.placeholder(tf.int32, shape=(None, vocab_size))
        #target = tf.placeholder(tf.int32, shape=(batch_size, class_size))
        input_vecs = tf.placeholder(tf.float32, shape=(max_seq_len, vocab_size))
        target = tf.placeholder(tf.float32, shape=(1, class_size))

        #input_vecs = tf.placeholder(tf.float32, shape=(batch_size, None, vocab_size))
        #target = tf.placeholder(tf.float32, shape=(batch_size, class_size))
        #inputs = tf.unpack(input_vecs, 500, axis=0)#equivalent to the line below
        inputs = tf.split(0, max_seq_len, input_vecs)# each row is an input
        #inputs = tf.transpose(input_vecs) 
        #inputs = tf.split(1, 500, inputs)

        cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)
        #init_state = cell.zero_state(1, dtype=tf.float32)
        outputs, last_state = tf.nn.rnn(cell, inputs, dtype=tf.float32)
        #outputs, last_state = tf.nn.dynamic_rnn(cell, input_vecs, dtype=tf.float32)

        #val = tf.transpose(outputs, [1, 0, 2])
        #last = tf.gather(val, int(val.get_shape()[0]) - 1)
        last_output = outputs[-1]

        #define predict, softmax
        weight = tf.Variable(tf.truncated_normal([lstm_hidden_size, int(target.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
        prediction = tf.nn.softmax(tf.matmul(last_output, weight) + bias) 

        #define loss
        cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
        loss = cross_entropy

        #define train
        optimizer = tf.train.AdamOptimizer()
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cross_entropy)

        '''
        #define loss
        target_one_hot = tf.one_hot(target, class_size, 1, 0)
        logits = tf.contrib.layers.fully_connected(last_state, class_size, activation_fn=None)
        loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

        #define train op
        train = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer='Adam', learning_rate=0.01)
        #optimizer = tf.train.GradientDescentOptimizer(0.01)
        #train = optimizer.minimize(loss) 
        '''
        #initilization
        init = tf.initialize_all_variables()
#        batch_size = 10
#        count = 0
#        batch_inputs = []
#        batch_outputs = []

        with tf.Session() as sess:
            sess.run(init)
            for pas in range(10):
                print('train======pas', pas)
                for i, doc in enumerate(train_data):
    #                count +=1
                    vecs = get_input_words_vec(doc.words)
    #                batch_inputs.append(vecs)
                    y = np.zeros(shape=(1, class_size))
                    y[0, doc.topic_id] = 1
    #                batch_outputs.append(y)
                    ret = sess.run([train, loss], feed_dict={input_vecs:vecs, target:y})
                    #print('doc', i)
                print('score in train, pas======', pas, ':', score(sess, prediction, input_vecs, train_data))
                print('score in test, pas======', pas, ':', score(sess, prediction, input_vecs, test_data))
                
                '''
                if count==batch_size:
                    pdb.set_trace()
                    sess.run([train, loss], feed_dict={input_vecs:batch_inputs, target:batch_outputs})
                    count = 0
                    batch_inputs = []
                    batch_outputs = []
                '''
def score2(sess, predict_op, input_vecs, test_docs):
    total = 0
    correct = 0
    count = 0
    batch_inputs = []
    batch_outputs = []
    for doc in test_docs:
        vecs = get_input_words_vec(doc.words)
        batch_inputs.append(vecs)
        batch_outputs.append(doc.topic_id)
        total +=1
        count +=1
        if count==batch_size:
            rets = sess.run([predict_op], feed_dict={input_vecs:batch_inputs, })
            rets = rets[0]
            preds = np.argmax(rets, axis=1)
            num_correct = len(np.where(np.equal(preds, batch_outputs)==True)[0])
            correct += num_correct 
            count = 0
            batch_inputs = []
            batch_outputs = []
    return correct/float(total) 

def run2():
    Log.info(None, '---------RUN 2-------')
    #max_seq_len = 500 #TODO change to dynamic sequence length

    build_vocab()

    with tf.Graph().as_default():
        Log.info(None, 'Structure the graph...')
        #define the model
        #input_vecs = tf.placeholder(tf.int32, shape=(None, vocab_size))
        #target = tf.placeholder(tf.int32, shape=(batch_size, class_size))
        #input_vecs = tf.placeholder(tf.float32, shape=(max_seq_len, vocab_size))
        #target = tf.placeholder(tf.float32, shape=(1, class_size))

        input_vecs = tf.placeholder(tf.float32, shape=(batch_size, max_seq_len, vocab_size))
        targets = tf.placeholder(tf.float32, shape=(batch_size, class_size))
        #inputs = tf.unpack(input_vecs, 500, axis=0)#equivalent to the line below
        #inputs = tf.split(0, max_seq_len, input_vecs)# each row is an input
        #inputs = tf.transpose(input_vecs) 
        #inputs = tf.split(1, 500, inputs)

        cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)
        #init_state = cell.zero_state(1, dtype=tf.float32)
        #outputs, last_state = tf.nn.rnn(cell, inputs, dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(cell, input_vecs, dtype=tf.float32)

        #val = tf.transpose(outputs, [1, 0, 2])
        #last = tf.gather(val, int(val.get_shape()[0]) - 1)
        last_outputs = outputs[:, -1, :]

        #define predict, softmax
        weight = tf.Variable(tf.truncated_normal([lstm_hidden_size, int(targets.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[targets.get_shape()[1]]))
        predictions = tf.nn.softmax(tf.matmul(last_outputs, weight) + bias) 

        #define loss
        cross_entropy = -tf.reduce_sum(targets * tf.log(predictions))
        loss = cross_entropy

        #define train
        optimizer = tf.train.AdamOptimizer()
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cross_entropy)

        '''
        #define loss
        target_one_hot = tf.one_hot(target, class_size, 1, 0)
        logits = tf.contrib.layers.fully_connected(last_state, class_size, activation_fn=None)
        loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

        #define train op
        train = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer='Adam', learning_rate=0.01)
        #optimizer = tf.train.GradientDescentOptimizer(0.01)
        #train = optimizer.minimize(loss) 
        '''
        #initilization
        init = tf.initialize_all_variables()
        count = 0
        batch_inputs = []
        batch_outputs = []

        with tf.Session() as sess:
            sess.run(init)
            print('score at init...')
            print('score in train at init:', score2(sess, predictions, input_vecs, train_data))
            for pas in range(10):
                print('train======pas', pas)
                for i, doc in enumerate(train_data):
                    count +=1
                    vecs = get_input_words_vec(doc.words)
                    batch_inputs.append(vecs)
                    #y = np.zeros(shape=(class_size, ))
                    y = [0]*class_size
                    y[doc.topic_id] = 1
                    batch_outputs.append(y)
                    #ret = sess.run([train, loss], feed_dict={input_vecs:vecs, target:y})
                
                    if count==batch_size:
                        print('-------------batch', (i+1)/batch_size)
                        sess.run([train, loss], feed_dict={input_vecs:batch_inputs, targets:batch_outputs})
                        count = 0
                        batch_inputs = []
                        batch_outputs = []
                print('score in train, pas======', pas, ':', score2(sess, predictions, input_vecs, train_data))
                print('score in test, pas======', pas, ':', score2(sess, predictions, input_vecs, test_data))

def main():
    #run1()
    run2()
    print('DONE')

if __name__ == '__main__':
    main() 
