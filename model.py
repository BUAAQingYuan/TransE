__author__ = 'PC-LiNing'

import tensorflow as tf
import data_helpers
import datetime
import load_data
import argparse
import numpy
import time
import evaluate

entity_size = 14951
relation_size = 1345
embedding_size = 50

# training
Test_size = 5000
Train_size = 54071
# Train_size = 483142
BATCH_SIZE = 16
EVAL_FREQUENCY = 1000
NUM_EPOCHS = 5

FLAGS = None


def train():
    # load data
    # train_data = [train_size,3]
    # test_data = [test_size,3]
    print("loading data...")
    test_data = load_data.load_train_test()
    train_data = test_data[:Train_size, :]
    test_data = test_data[Train_size:, :]

    train_data_node = tf.placeholder(tf.int32, shape=(None, 3))
    train_neg_node = tf.placeholder(tf.int32, shape=(None, 2*(entity_size-1), 3))

    test_scores_node = tf.placeholder(tf.float32, shape=(Test_size, entity_size))
    test_labels_node = tf.placeholder(tf.int32, shape=(Test_size,))

    entity_embedding = tf.Variable(tf.random_uniform([entity_size, embedding_size], -1.0, 1.0), name="entity_embedding")
    relation_embedding = tf.Variable(tf.random_uniform([relation_size, embedding_size], -1.0, 1.0), name="relation_embedding")

    # inputs = [batch_size,3]
    # neg_inputs = [batch_size,2*(entity_size-1),3]
    def model(inputs, neg_inputs):
        # [batch_size]
        inputs_h = inputs[:, 0]
        inputs_t = inputs[:, 1]
        inputs_r = inputs[:, 2]
        # [batch_size,2*(entity_size-1)]
        neg_inputs_h = neg_inputs[:, :, 0]
        neg_inputs_t = neg_inputs[:, :, 1]
        neg_inputs_r = neg_inputs[:, :, 2]
        # [batch_size,embedding_size]
        h_embed = tf.nn.embedding_lookup(entity_embedding, inputs_h)
        t_embed = tf.nn.embedding_lookup(entity_embedding, inputs_t)
        r_embed = tf.nn.embedding_lookup(relation_embedding, inputs_r)
        # [batch_size , 2*(entity_size-1),embedding_size]
        h_neg = tf.nn.embedding_lookup(entity_embedding, neg_inputs_h)
        t_neg = tf.nn.embedding_lookup(entity_embedding, neg_inputs_t)
        r_neg = tf.nn.embedding_lookup(relation_embedding, neg_inputs_r)
        # [batch_size,1]
        delta = tf.reduce_sum((h_embed + r_embed - t_embed)**2, 1, keep_dims=True)
        # neg delta = [batch_size,2*(entity_size-1)]
        neg_delta = tf.reduce_sum((h_neg + t_neg - r_neg)**2, 2)
        # neg delta = [batch_size,1], equals to div (2*entity_size-2)
        neg_delta = tf.reduce_mean(neg_delta, 1, keep_dims=True)
        return delta, neg_delta

    pos_one, neg_one = model(train_data_node, train_neg_node)
    margin = 0.0
    # loss = tf.reduce_mean(tf.maximum(pos_one + margin - neg_one, 0))
    loss = tf.reduce_mean(pos_one + margin - neg_one)

    # predict
    # test_inputs = [batch_size,3]
    def get_embeddings(test_inputs):
        inputs_h = test_inputs[:, 0]
        inputs_t = test_inputs[:, 1]
        # labels = [batch_size]
        inputs_r = test_inputs[:, 2]
        # [batch_size,embedding_size]
        h_embed = tf.nn.embedding_lookup(entity_embedding, inputs_h)
        t_embed = tf.nn.embedding_lookup(entity_embedding, inputs_t)
        r_embed = tf.nn.embedding_lookup(relation_embedding, inputs_r)
        return h_embed, t_embed, r_embed

    def evalution(scores,labels):
        # get top k
        # scores = [Test_size,entity_size]
        # labels = [Test_size]
        # h_result = [Test_size]
        h_result = tf.nn.in_top_k(scores, labels, k=10)
        # acc
        h_acc = tf.reduce_mean(tf.cast(h_result, tf.float32))
        return h_acc

    h_embed, t_embed, r_embed = get_embeddings(train_data_node)
    acc = evalution(test_scores_node, test_labels_node)

    # train
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # runing the training
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print('Initialized!')
        # generate batches
        batches = data_helpers.batch_iter(list(zip(train_data)),BATCH_SIZE,NUM_EPOCHS)
        # batch count
        batch_count = 0
        epoch = 1
        print("Epoch "+str(epoch)+":")
        for batch in batches:
            batch_count += 1
            # train process
            x_batch = numpy.squeeze(batch)
            # generate neg data
            neg_x_batch = load_data.generate_neg_data(x_batch)
            feed_dict = {train_data_node: x_batch, train_neg_node: neg_x_batch}
            _,step,losses = sess.run([train_op, global_step,loss],feed_dict=feed_dict)
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{}: step {}, loss {:g}".format(time_str, step, losses))

            # test process
            if float((batch_count * BATCH_SIZE) / Train_size) > epoch:
                epoch += 1
                print("Epoch "+str(epoch)+":")
            if batch_count % EVAL_FREQUENCY == 0:
                # get test scores
                feed_dict = {train_data_node: test_data}
                # get embedding
                print("get embedding...")
                h_embedding, t_embedding, r_embedding, entity_embed = sess.run([h_embed, t_embed, r_embed, entity_embedding], feed_dict=feed_dict)
                # compute score
                t_start = time.time()
                h_acc, t_acc, h_mean_rank, t_mean_rank = evaluate.compute_acc(h_embedding, t_embedding, r_embedding, entity_embed)
                t_end = time.time()
                t = t_end-t_start
                print("computing acc..., cost :%s" % t)
                hit_acc = (h_acc + t_acc)/2.0
                mean_rank = int((h_mean_rank + t_mean_rank)/2)
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("{}: step {}, h-acc {:g}, t-acc {:g}, Hit@10 {:g}, h_rank {}, t_rank {}, mean_rank {}".format(
                    time_str, step, h_acc, t_acc, hit_acc, h_mean_rank, t_mean_rank, mean_rank))
                print("\n")


def main(_):
    # if tf.gfile.Exists(FLAGS.summaries_dir):
    #    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    # tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='/tmp/kb2e',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()

