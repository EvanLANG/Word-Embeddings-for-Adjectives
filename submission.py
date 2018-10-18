## Submission.py for COMP6714-Project2
###################################################################################################################
import re
import os
import sys
import math
import random
import datetime
import zipfile
import collections
import numpy as np
import tensorflow as tf
import spacy
import gensim
import pickle
from gensim import utils

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#Hyper Parameters
batch_size = 128
skip_window = 2
num_samples = 4
vocabulary_size = 10000 #15000
learning_rate = 0.002 #0.003
num_sampled = 300 #1700number of negative samples
embeddings_file_name = "adjective_embeddings.txt"
num_steps = 100001
embedding_dim = 200

#top_k = 8
sample_window = 100
sample_size= 20
sample_examples = np.array(random.sample(range(sample_window), sample_size))
parser = spacy.load('en')
data_index = skip_window

def skipto(count, reverse_dictionary, data, total_size):
    global data_index
    token = data[data_index]
    tem = count[token]/total_size
    if tem != 0:
        return random.random() > (math.sqrt(tem/0.001) + 1) * 0.001 / tem
    else:
        return False

def generate_batch(data, count, dictionary, reverse_dictionary, adjective, batch_size, num_samples, skip_window):
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    for i in range(batch_size // num_samples):
        word_to_use = list()
        for ii in range(data_index-skip_window, data_index+skip_window+1):
            if ii != data_index:
                word_to_use.append(data[ii])
        random.shuffle(word_to_use)
        center_word = data[data_index]
        for j in range(num_samples):
            batch[i * num_samples + j] = center_word
            context_word = word_to_use.pop()
            labels[i * num_samples + j, 0] = context_word
        data_index += 1
        total_size = len(data)
        while (data_index + skip_window >= total_size) or (skipto(count, reverse_dictionary, data, total_size)):
            if data_index + skip_window >= total_size:
                data_index = skip_window
            else:
                data_index += 1
    return batch, labels


def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    if os.path.exists(embeddings_file_name):
        return
    with open(data_file, 'rb') as f:
        data, count, dictionary, reverse_dictionary, adjective = pickle.loads(f.read())
    adjective = [i for i in adjective if i in dictionary]

    # Specification of test Sample:
    sample_size = 20       # Random sample of words to evaluate similarity.
    sample_window = 100    # Only pick samples in the head of the distribution.
    sample_examples = np.random.choice(sample_window, sample_size, replace=False) # Randomly pick a sample of size 16

    ## Constructing the graph...
    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('Inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        with tf.name_scope('Embeddings'):
            sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dim], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dim], stddev=1.0 / math.sqrt(embedding_dim)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

        with tf.name_scope('Adam'):
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

        with tf.name_scope('Normalization'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
        sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
        similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)

        #create a summary to monitor the cost tensor
        tf.summary.scalar("cost", loss)
        merged_summary_op = tf.summary.merge_all()


    with tf.Session(graph=graph) as session:
        # nitialize all variables 
        session.run(tf.global_variables_initializer())
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(data, count, dictionary, reverse_dictionary, adjective, batch_size, num_samples, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)
            average_loss += loss_val
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d is: %f' % (step, average_loss))
                    average_loss = 0
            # Evaluate similarity after every 10000 iterations.
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(sample_size):
                    sample_word = reverse_dictionary[sample_examples[i]]
                    top_k = 8 
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % sample_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    # print(log)
                # print()
        final_embeddings = normalized_embeddings.eval()
        #np.save(embeddings_file_name, final_embeddings)
        with open(embeddings_file_name, 'w') as model:
            print(len(adjective), embedding_dim, file = model)
            for w in range(len(adjective)):
                print(adjective[w], end=' ', file=model)
                cur_word = dictionary[adjective[w]]
                line = []
                for col in range(embedding_dim):
                    line.append(str(final_embeddings[cur_word][col]))
                print(' '.join(line) , file=model)


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    count = {dictionary[k]:v for k,v in count}
    return data, count, dictionary, reversed_dictionary

def build_token(file):
    file = parser(file)
    adjective = set()
    data = []
    for i in file:
        if i.pos_ == 'ADJ':
            adjective.add(i.text.lower())
        if i.text.isalpha():
            data.append(i.text.lower())
    return data, adjective

def process_data(input_data):
    filename = "processed_data"
    #print("READING DATA")
    data = []
    adjective = set()
    with zipfile.ZipFile(input_data) as f:
        for file in f.namelist():
            doc = tf.compat.as_str(f.read(file)).strip()            
            if doc:
                words, adjs = build_token(doc)
                data.extend(words)
                for w in adjs:
                    adjective.add(w)
    # total_size: 843595                  
    # print("Data size is: %d" % len(data))
    # print(data[:5])    
        
    if not os.path.exists(filename):
        #print("SAVING DATA")
        with open(filename, 'wb') as f:
            f.write(pickle.dumps((*build_dataset(data, vocabulary_size), adjective)))
    return filename 



def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary = False)
    output = model.most_similar(input_adjective, topn = top_k)
    temp=[]
    for c in output:
        temp.append(c[0])    
    return temp


# if __name__ == "__main__":
    # input_dir = './BBC_Data.zip'
    # data_file=process_data(input_dir)
    #adjective_embeddings("processed_data", embeddings_file_name, num_steps, embedding_dim)



