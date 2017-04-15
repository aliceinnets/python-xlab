'''
Created on 14 Apr 2017

@author: alice<aliceinnets[at]gmail.com>
'''

from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow as tf
import util.oneliners as oneliners

def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def fc_layer(input, channels_in, channels_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def model(learning_rate, use_two_fully_connected_layers, use_two_conv_layers, run_name):
    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(oneliners.test_results_path + 'mnist_data/', one_hot=True)
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image("input", x_image, 3)
    
    if use_two_conv_layers:
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_out = conv_layer(conv1, 32, 64, "conv2")
    else:
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
    flattened = tf.reshape(conv_out, [-1, 7*7*64])
    
    if use_two_fully_connected_layers:
        fc1 = fc_layer(flattened, 7*7*64, 1024, "fc1")
        embedding_input = fc1
        embedding_size = 1024
        logits = fc_layer(fc1, 1024, 10, "fc1")
    else:
        embedding_input = flattened
        embedding_size = 7*7*64
        logits = fc_layer(flattened, 7*7*64, 10, "fc1")
        
    
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
#         cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar("xent", cross_entropy)
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    
    merged_summary = tf.summary.merge_all()
    
    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(oneliners.test_results_path+"mnist_demo/"+run_name+"/")
    writer.add_graph(sess.graph)
    
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = oneliners.test_results_path+"mnist_demo/"+run_name+"/" + 'sprite_1024.png'
    embedding_config.metadata_path = oneliners.test_results_path+"mnist_demo/"+run_name+"/" + 'labels_1024.tsv'
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, merged_summary], feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)
            print("step %d, training accuracy %g" % (i, train_accuracy))
        
        if i % 100 == 0:
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
            saver.save(sess, os.path.join(oneliners.test_results_path+"mnist_demo/"+run_name+"/", "model.ckpt"), i)
        
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
    
    sess.close()
    
    
def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)
  
if __name__ == '__main__':
    model(1E-3, True, True, "11")