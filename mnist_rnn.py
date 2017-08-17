# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot=True)


learning_rate = 0.0001
batch_size = 100
n_iterations = 25 * mnist.train.num_examples / batch_size

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

def RNN(x, n_layers=1):
	# x shape: (batch_size, n_steps, n_input)
	# desired shape: list of n_steps with element shape (batch_size, n_input)
	x = tf.transpose(x, [1, 0, 2])   # [n_steps, batch_size, n_input]
	x = tf.reshape(x, [-1, n_input])   # [n_steps * batch_size, n_input]
	x = tf.split(x, n_steps, 0)   # n_steps * [batch_size, n_input]
	if n_layers == 1:
		lstm = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
		# from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import DropoutWrapper
		# lstm = DropoutWrapper(lstm, output_keep_prob=0.5)
	else:
		lstm = rnn.MultiRNNCell(
			[rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True) for _ in range(n_layers)])
	output, state = rnn.static_rnn(lstm, x, dtype=tf.float32)
	# Classifier weights and biases
	weight = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
	biases = tf.Variable(tf.zeros([n_classes]))
	final = tf.matmul(output[-1], weight) + biases
	return final

def RNN_DIY(x, state, output, n_steps, n_input, n_hidden, n_classes):
	# Parameters:
	# Input gate: input, previous output, and bias
	ix = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
	im = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	ib = tf.Variable(tf.zeros([1, n_hidden]))
	# Forget gate: input, previous output, and bias
	fx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
	fm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	fb = tf.Variable(tf.zeros([1, n_hidden]))
	# Memory cell: input, state, and bias
	cx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
	cm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	cb = tf.Variable(tf.zeros([1, n_hidden]))
	# Output gate: input, previous output, and bias
	ox = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
	om = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	ob = tf.Variable(tf.zeros([1, n_hidden]))
	# Classifier weights and biases
	w = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))

	# Definition of the cell computation
	def lstm_cell(i, o, state):
		input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
		forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
		update = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
		state = forget_gate * state + input_gate * update
		output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
		return output_gate * tf.tanh(state), state

	# Unrolled LSTM loop
	outputs = list()

	# x shape: (batch_size, n_steps, n_input)
	# desired shape: list of n_steps with element shape (batch_size, n_input)
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(x, n_steps, 0)
	for i in x:
		output, state = lstm_cell(i, output, state)
		outputs.append(output)
	logits = tf.matmul(outputs[-1], w) + b
	return logits

state = tf.placeholder(tf.float32, [None, n_hidden])
output = tf.placeholder(tf.float32, [None, n_hidden])
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# pred = RNN_DIY(x, state, output, n_steps, n_input, n_hidden, n_classes)   # 0.9816
pred = RNN(x, 1)   # 0.9804
# pred = RNN(x, 2)   # 0.9809

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
# Launch the graph
sess.run(init)
_state = numpy.zeros([batch_size, n_hidden])
_output = numpy.zeros([batch_size, n_hidden])
for step in range(n_iterations):
	batch_x, batch_y = mnist.train.next_batch(batch_size)
	batch_x = batch_x.reshape((batch_size, n_steps, n_input))
	sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, state: _state, output: _output})

	if step % 500 == 0:
		acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, state: _state, output: _output})
		loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, state: _state, output: _output})
		print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(
			loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
print "Optimization Finished!"

test_number = 0
test_accuracy = 0
total = mnist.test.num_examples
while test_number < total:
	test_x = mnist.test.images[test_number: min(test_number + 1000, total)]
	test_x = test_x.reshape((-1, n_steps, n_input))
	test_y = mnist.test.labels[test_number: min(test_number + 1000, total)]
	test_batch = test_x.shape[0]
	_state = numpy.zeros([test_batch, n_hidden])
	_output = numpy.zeros([test_batch, n_hidden])
	test_accuracy += sess.run(accuracy, feed_dict={x: test_x, y: test_y, state: _state, output: _output})\
					 * test_batch
	test_number += test_batch
test_accuracy /= mnist.test.num_examples
print "Testing Accuracy:", test_accuracy
sess.close()


'''
BasicLSTMCell __doc__
Basic LSTM recurrent network cell.
The implementation is based on: http://arxiv.org/abs/1409.2329.
We add forget_bias (default: 1) to the biases of the forget gate in order to
reduce the scale of forgetting in the beginning of the training.
It does not allow cell clipping, a projection layer, and does not
use peep-hole connections: it is the basic baseline.
For advanced models, please use the full LSTMCell that follows.

MultiRNNCell __doc__
RNN cell composed sequentially of multiple simple cells.
'''

