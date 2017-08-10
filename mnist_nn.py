# -*- coding:utf-8 -*-

import fullyConnected
import tensorflow as tf
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

print 'train_examples', mnist.train.num_examples
print 'test_examples', mnist.test.num_examples
# print mnist.test.images
# print mnist.test.labels

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# Graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None])

logits = fullyConnected.inference(x, 100, 100)
loss = fullyConnected.loss(logits, y)
train = fullyConnected.training(loss, learning_rate)
evaluation = fullyConnected.evaluation(logits, y)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([train, loss], feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		# Display
		if (epoch + 1) % display_step == 0:
			print 'Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost)

	print 'Optimization Finished'

	# Test model
	accuracy = sess.run(evaluation, feed_dict={x: mnist.test.images, y: mnist.test.labels})
	print accuracy
	print mnist.test.num_examples
	print 'Accuracy:', float(accuracy) / mnist.test.num_examples   # 0.9535

