# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print 'train_examples', mnist.train.num_examples
print 'test_examples', mnist.test.num_examples

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# filter: [filter_height, filter_width, in_channels(=input in_channels), out_channels(feature maps)]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# input: [batch(adapted by data), in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# output: [batch, height, width, out_channels]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Now image size is reduced to 7*7
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# softmax: p(i) = e^xi / sum(e^x)
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# y_ & y_conv: [batch, 10]
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# Adam algorithm: ...
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=[])
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())

for i in range(25 * mnist.train.num_examples / 100):
	batch = mnist.train.next_batch(100)
	if i % 100 == 0:
		# batch[0], batch[1]: numpy.ndarray
		train_accuracy = accuracy.eval(feed_dict={
			x: batch[0], y_: batch[1], keep_prob: 1.0})
		print "step %d, training accuracy %.3f" % (i, train_accuracy)
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	# result = tensor.eval(feed_dict)
	# result = operation.run(feed_dict)
	# [results...] = session.run([tensors...], feed_dict)

print "Training finished"

test_number = 0
test_accuracy = 0
total = mnist.test.num_examples
while test_number < total:
	test_x = mnist.test.images[test_number: min(test_number + 1000, total)]
	test_y = mnist.test.labels[test_number: min(test_number + 1000, total)]
	test_batch = test_x.shape[0]
	test_accuracy += accuracy.eval(feed_dict={
		x: test_x, y_: test_y, keep_prob: 1.0}) * test_batch
	test_number += test_batch
test_accuracy /= mnist.test.num_examples

print "test accuracy %.3f" % test_accuracy   # 0.9930
sess.close()


'''
conv2d __doc__
Computes a 2-D convolution given 4-D `input` and `filter` tensors.
Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:
1. Flattens the filter to a 2-D matrix with shape
`[filter_height * filter_width * in_channels, output_channels]`.
2. Extracts image patches from the input tensor to form a *virtual*
tensor of shape `[batch, out_height, out_width,
filter_height * filter_width * in_channels]`.
3. For each patch, right-multiplies the filter matrix and the image patch
vector.
In detail, with the default NHWC format,
output[b, i, j, k] =
sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
filter[di, dj, q, k]
Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
Args:
input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
A 4-D tensor. The dimension order is interpreted according to the value
of `data_format`, see below for details.
filter: A `Tensor`. Must have the same type as `input`.
A 4-D tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`
strides: A list of `ints`.
1-D tensor of length 4.  The stride of the sliding window for each
dimension of `input`. The dimension order is determined by the value of
`data_format`, see below for details.
padding: A `string` from: `"SAME", "VALID"`.
The type of padding algorithm to use.
use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
Specify the data format of the input and output data. With the
default format "NHWC", the data is stored in the order of:
[batch, height, width, channels].
Alternatively, the format could be "NCHW", the data storage order of:
[batch, channels, height, width].
name: A name for the operation (optional).
Returns:
A `Tensor`. Has the same type as `input`.
A 4-D tensor. The dimension order is determined by the value of
`data_format`, see below for details.

dropout __doc__
Computes dropout.
With probability `keep_prob`, outputs the input element scaled up by
`1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
sum is unchanged.
By default, each element is kept or dropped independently.  If `noise_shape`
is specified, it must be
[broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
kept independently and each row and column will be kept or not kept together.
Args:
x: A tensor.
keep_prob: A scalar `Tensor` with the same type as x. The probability
that each element is kept.
noise_shape: A 1-D `Tensor` of type `int32`, representing the
shape for randomly generated keep/drop flags.
seed: A Python integer. Used to create random seeds. See
@{tf.set_random_seed}
for behavior.
name: A name for this operation (optional).
Returns:
A Tensor of the same shape of `x`.
Raises:
ValueError: If `keep_prob` is not in `(0, 1]`.

conv2d_transpose __doc__
The transpose of `conv2d`.
  This operation is sometimes called "deconvolution" after [Deconvolutional
  Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf), but is
  actually the transpose (gradient) of `conv2d` rather than an actual
  deconvolution.
  Args:
    value: A 4-D `Tensor` of type `float` and shape
      `[batch, height, width, in_channels]` for `NHWC` data format or
      `[batch, in_channels, height, width]` for `NCHW` data format.
    filter: A 4-D `Tensor` with the same type as `value` and shape
      `[height, width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: A list of ints. The stride of the sliding window for each
      dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the @{tf.nn.convolution$comment here}
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the returned tensor.
  Returns:
    A `Tensor` with the same type as `value`.
  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.
'''

