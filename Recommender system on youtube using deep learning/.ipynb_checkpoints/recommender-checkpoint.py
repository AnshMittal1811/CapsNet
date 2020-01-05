from __future__ import print_function
import sys
import linecache
import numpy as np
import tensorflow as tf
import time
import math
import random

device = '/gpu:0'

# Insert test and train path and an initial file for
# init_file = "path to initial csv file"
# train = "path to train dataset"
# test = "path to test dataset"
l_dict = {}
s_dict = {}

# Maximum window size and embeddings size
max_win_size = 100
emb_size = 50

# Network Parameters
n_hidden_1 = 50 # 1st layer number of features
# n_hidden_2 = 256 # 2nd layer number of features

# Learning rate, epochs, steps
lr = 0.001
epochs = 100
dstep = 1


def init_dictionary_data(dict_file):
    #s {sku} 0 is used for padding embedding
    #l {label} must be sorted by descending order
    label_cnt = 0
    sku_cnt = 1
    
    # Opening the dictionary contains 
    f = open(dict_file,'r')
    
    for line in f:
        
        line = line.strip().split('\t')
        
        i = line[0]
        
        if '__label__' + i not in l_dict:
            l_dict['__label__' + i] = label_cnt
            label_cnt += 1
            
        if i not in s_dict:
            s_dict[i] = sku_cnt
            sku_cnt += 1

            

def init_videodata(read_file):
    #0 is used for padding embedding
    
    sku_cnt = 1
    label_cnt = 0
    
    f = open(read_file,'r')
    for line in f:
        line = line.strip().split(' ')
        for i in line:
            if i.find('__label__') == 0:
                if i not in l_dict:
                    l_dict[i] = label_cnt
                    label_cnt += 1;
            else:
                if i not in s_dict:
                    s_dict[i] = sku_cnt
                    sku_cnt += 1;

                    
def readData(batch):
    batch_size = len(batch)
    x = np.zeros((batch_size, max_win_size))
    mask = np.zeros((batch_size, max_win_size))
    y = []
    word_num = np.zeros((batch_size))
    line_no = 0
    for line in batch:
        line = line.strip().split(' ')
        label_lst = []
        col_no = 0
        for i in line:
            if i in l_dict:
                label_lst.append(l_dict[i])
            elif i in s_dict:
                x[line_no][col_no] = s_dict[i]
                mask[line_no][col_no] = 1
                col_no += 1
                if col_no >= max_win_size:
                    break
        y.append(label_lst[random.randint(0, len(label_lst)-1)])
        word_num[line_no] = col_no
        line_no += 1

    return x, np.array(y).reshape(batch_size, 1), mask.reshape(batch_size, max_win_size, 1), word_num.reshape(batch_size, 1)




# Driver code
def main():
    init_dictionary_data(init_file)
    init_videodata(train)
    
    n_input = len(s_dict) + 1
    n_classes = len(l_dict) 
    
#     #train_lst = linecache.getlines(train)
#     print("Class Num: ", n_classes)
#     print("Vocab Num: ", n_input)
    
    
    
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([emb_size, n_hidden_1]))
        # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        #'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]))
        # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        #'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    
    
    # Create Neural Network model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        #x = tf.nn.dropout(x, 0.5)
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        dlayer_1 = tf.nn.dropout(layer_1, 0.4)
        layer_2 = tf.add(tf.matmul(dlayer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        #Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer
        #return layer_1
    
    
    
    embedding = {
        'input':tf.Variable(tf.random_uniform([n_input, emb_size], -1.0, 1.0))
        # 'output':tf.Variable(tf.random_uniform([len(l_dict)+1, emb_size], -1.0, 1.0))
    }
    
    
    
    emb_mask = tf.placeholder(tf.float32, shape=[None, max_win_size, 1])
    word_num = tf.placeholder(tf.float32, shape=[None, 1])
    
    x_batch = tf.placeholder(tf.int32, shape=[None, max_win_size])
    y_batch = tf.placeholder(tf.int64, [None, 1])

    
    input_embedding = tf.nn.embedding_lookup(embedding['input'], x_batch)
    project_embedding = tf.div(tf.reduce_sum(tf.multiply(input_embedding,emb_mask), 1),word_num)
    
    # Construct model and store into object
    predictions = multilayer_perceptron(project_embedding, weights, biases)
    
    # Construct the variables for the NCE (Noise Contrastive Estimation) loss
    nce_weights = tf.Variable(tf.truncated_normal([n_classes, n_hidden_1],stddev=1.0 / math.sqrt(n_hidden_1)))
    nce_biases = tf.Variable(tf.zeros([n_classes]))
    
    out_layer = tf.matmul(predictions, nce_weights, transpose_b=True) + nce_biases
    cost = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=y_batch,
                         inputs=predictions,
                         num_sampled=10,
                         num_classes=n_classes))
    
    # Optimizer can be changed according to the number of videos
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

    # initializing tensorflow global variable
    init = tf.global_variables_initializer()
    
    # setting the configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    
    # Tensorflow session
    with tf.Session(config=config) as sess:
        
        sess.run(init)
        
        # Training cycle
        start_time = time.time()
        for epoch in range(epochs):
            f_train = open(train, 'r')
            avg_cost = 0.
            i = 0
            
            while 1:
                
                x, y, batch_mask, word_no = readData(f_train.readlines(10000))
                
                if y.shape[0] == 0:
                    break
                    
                i += 1
                _,c = sess.run([optimizer, cost], feed_dict={x_batch: x, emb_mask: batch_mask, word_num: word_no, y_batch: y})
                
                if i % 10000 == 0:
                    print("Epoch %d Batch %d Elapsed time %fs Batchsize %d,%d avg_cost_per_batch %f" %(epoch, i, time.time() - start_time ,x.shape[0], y.shape[0], avg_cost/i))
                    
                avg_cost += c
    
            # Display logs per epoch step
            if epoch % dstep == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost/i))
                
            if avg_cost/i < 0.003:
                break
        
        # Testing the model
        f_test = open(test,'r')
        final_acc = 0
        i = 0
        
        while 1:
            l = f_test.readlines(1000)
            
            if len(l) == 0:
                break
                
            x, y, batch_mask, word_no = readData(l)
            i += 1
            
            correct_pred = tf.equal(tf.argmax(out_layer, 1), tf.reshape(y_batch, [y.shape[0]]))
            
            
            # Calculate accuracy
            acc = tf.reduce_mean(tf.cast(correct_pred, "float"))
            
            [batch_acc, pre_sub, true_sub] = sess.run([acc, tf.argmax(out_layer, 1), 
                                                       tf.reshape(y_batch, [y.shape[0]])], 
                                                      feed_dict = {x_batch: x, y_batch: y, emb_mask: batch_mask, word_num: word_no})
            
            print(pre_sub, true_sub)
            final_acc += batch_acc
            
            # Ctrl + Tab to print the accuracy of the recommendations from the ground truth
#             print(batch_acc)
#             print(i, " Final Accuracy: ", final_acc * 1.0 / i)


# Calling driver using Tensorflow-GPU
# Download tensorflow-gpu to anaconda
if __name__ == '__main__':
    with tf.device(device):
        main()