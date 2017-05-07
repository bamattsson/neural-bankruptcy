import numpy as np
import tensorflow as tf

from algorithm import Algorithm


class MultiLayerPerceptron(Algorithm):

    def __init__(self, n_input, n_hidden_1, n_hidden_2, num_epochs):
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2

        self.n_class = 2

        # Create TF graph and session
        tf.reset_default_graph()
        self.graph_nodes = self._get_graph()

        sess = tf.Session()
        sess.run([tf.global_variables_initializer()])
        # TODO: add early stopping with tf.train.Saver

    def fit(self, samples, labels):
        pass  # TODO

    def predict(self, samples):
        pass  # TODO

    def _get_graph(self):
        # Create placeholders for input
        x_input = tf.placeholder(tf.float32, shape=(None, self.n_input))
        y_input = tf.placeholder(tf.int32, shape=(None))

        # Variables
        W_layer_1 = tf.Variable(tf.truncated_normal((self.n_input, self.n_hidden_1), stddev=0.1), name='W_1_layer')
        b_layer_1 = tf.Variable(tf.truncated_normal((self.n_hidden_1), stddev=0.1), name='b_1_layer')
        W_layer_2 = tf.Variable(tf.truncated_normal((self.n_hidden_1, self.n_hidden_2), stddev=0.1), name='W_2_layer')
        b_layer_2 = tf.Variable(tf.truncated_normal((self.n_hidden_2), stddev=0.1), name='b_2_layer')
        W_layer_o = tf.Variable(tf.truncated_normal((self.n_hidden_2, self.n_class), stddev=0.1), name='W_out_layer')
        b_layer_o = tf.Variable(tf.truncated_normal((self.n_class), stddev=0.1), name='b_out_layer')

        # Model
        # TODO: we could make number of layers a hyperparameter pretty easily
        layer_1 = tf.add(tf.matmul(x_input, W_layer_1), b_layer_1)
        layer_1 = tf.nn.relu(layer_1)  # TODO: make this optional

        layer_2 = tf.add(tf.matmul(layer_1, W_layer_2), b_layer_2)
        layer_2 = tf.nn.relu(layer_2)  # TODO: make this optional

        logits = tf.add(tf.matmul(layer_2, W_layer_o), b_layer_o)
        prob = tf.nn.softmax(logits)

        # Loss and Accuracy
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=logits))
        correct_predictions = tf.equal(tf.argmax(logits, 1), y_input)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

        # Train operation
        optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # Save important nodes to dict and return
        graph = {'x_input': x_input,
                'y_input': y_input,
                'prob': prob,
                'loss': loss,
                'accuracy': accuracy,
                'optimize': optimize}

        return graph


def _oversampling_batch_iter(samples, labels, num_epochs, batch_size):
    """Batch iterator that oversamples the rare class so that both classes become equally frequent."""
    pos_examples = (labels == 1)
    pos_samples, pos_labels = samples[pos_examples], labels[pos_examples]
    neg_examples = np.logical_not(pos_examples)
    neg_samples, neg_labels = samples[neg_examples], labels[neg_examples]

    neg_batch_size = np.floor(batch_size / 2)
    pos_batch_size = np.ceil(batch_size / 2)

    neg_batch_iter = _batch_iter(neg_samples, neg_labels, num_epochs, neg_batch_size)
    pos_batch_iter = _batch_iter(pos_samples, pos_labels, num_epochs, pos_batch_size)

    for neg_batch, pos_batch in zip(neg_batch_iter, pos_batch_iter):
        neg_batch_samples, neg_batch_labels = neg_batch
        pos_batch_samples, pos_batch_labels = pos_batch
        batch_samples = np.concatenate((neg_batch_samples, pos_batch_samples), axis=0)
        batch_labels = np.concatenate((neg_batch_labels, pos_batch_labels), axis=0)
        yield batch_samples, batch_labels


def _batch_iter(samples, labels, num_epochs, batch_size):
    """A batch iterator that generates batches from the data."""
    data_size = len(labels)
    n_batches_per_epoch = int(np.ceil(data_size / batch_size))
    for epoch in range(num_epochs):
        indices = np.arange(data_size)
        for batch_num in range(n_batches_per_epoch):
            start_index = int(batch_num * batch_size)
            end_index = int(min((batch_num + 1) * batch_size, data_size))
            yield samples[indices[start_index:end_index]], labels[indices[start_index:end_index]],
