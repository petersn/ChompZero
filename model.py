#!/usr/bin/python

import os
import numpy as np
import tensorflow as tf

#DTYPE = tf.float16
DTYPE = tf.float32

FEATURE_COUNT = 2
BOARD_SIZE = 16

class Network:
	FILTERS = 32
	CONV_SIZE = 3
	BLOCK_COUNT = 8
	FINAL_FILTERS = 2
	OUTPUT_SOFTMAX_COUNT = BOARD_SIZE * BOARD_SIZE

	def __init__(self, scope_name, build_training=False):
		with tf.variable_scope(scope_name):
			self.build_graph()
			if build_training:
				with tf.variable_scope("training"):
					self.build_training()
		self.total_parameters = sum(np.product(var.shape) for var in tf.trainable_variables())

	def build_graph(self):
		self.input_ph = tf.placeholder(
			dtype=DTYPE,
			shape=[None, BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT],
			name="input_ph",
		)
		self.desired_policy_ph = tf.placeholder(
			dtype=DTYPE,
			shape=[None, BOARD_SIZE, BOARD_SIZE],
			name="desired_policy_ph",
		)
		self.desired_value_ph = tf.placeholder(
			dtype=DTYPE,
			shape=[None, 1],
			name="desired_value_ph",
		)
		self.learning_rate_ph = tf.placeholder(
			dtype=DTYPE,
			shape=[],
			name="learning_rate_ph",
		)
		self.is_training_ph = tf.placeholder(
			dtype=tf.bool,
			shape=[],
			name="is_training_ph",
		)

		#tf.summary.image("desired_policy", self.desired_policy_ph, 1)

		self.flow = self.input_ph
		# Stack an initial convolution.
		with tf.variable_scope("promote"):
			self.stack_convolution(3, FEATURE_COUNT, self.FILTERS)
			self.stack_nonlinearity()
		# Stack some number of residual blocks.
		for i in xrange(self.BLOCK_COUNT):
			with tf.variable_scope("block%i" % i):
				self.stack_block()
		# Stack a final batch-unnormalized 1x1 convolution down to some given number of filters.
		# The first feature is the policy layer, while the remaining are features for value head computation.
		self.stack_convolution(1, self.FILTERS, self.FINAL_FILTERS, batch_normalization=False)
		#assert self.flow.shape == [None, BOARD_SIZE, BOARD_SIZE, self.FINAL_FILTERS]
		self.policy_output = tf.reshape(self.flow[:,:,:,0], [-1, BOARD_SIZE, BOARD_SIZE])

		self.build_value_head()

	def build_value_head(self):
		with tf.variable_scope("value_head"):
			x = tf.reshape(self.flow, [-1, BOARD_SIZE * BOARD_SIZE * self.FINAL_FILTERS])
			x = tf.layers.dense(x, 32, activation="relu")
			self.value_output = tf.layers.dense(x, 1, activation="tanh")

	def build_training(self):
		# Make policy head loss.
		self.flattened = tf.reshape(self.policy_output, [-1, self.OUTPUT_SOFTMAX_COUNT])
		self.flattened_desired_output = tf.reshape(self.desired_policy_ph, [-1, self.OUTPUT_SOFTMAX_COUNT])
		self.policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
			labels=self.flattened_desired_output,
			logits=self.flattened,
		))

		tf.summary.scalar("policy_loss", self.policy_loss)

		# Make value head loss.
		self.value_loss = tf.reduce_mean(tf.square(self.desired_value_ph - self.value_output))
		tf.summary.scalar("value_loss", self.value_loss)

		# Make regularization loss.
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
		reg_variables = tf.trainable_variables()
		self.regularization_loss = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

		tf.summary.scalar("regularization_loss", self.regularization_loss)

		# Loss is the sum of these three.
		self.loss = 1.0 * self.policy_loss + self.value_loss + self.regularization_loss

		correct = tf.equal(
			tf.argmax(self.flattened, 1),
			tf.argmax(self.flattened_desired_output, 1),
		)
		self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		tf.summary.scalar("accuracy", self.accuracy)
		tf.summary.scalar("learning_rate", self.learning_rate_ph)

		# Associate batch normalization with training.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_step = tf.train.MomentumOptimizer(
				learning_rate=self.learning_rate_ph,
				momentum=0.9,
			).minimize(self.loss)
#			self.train_step = tf.train.AdamOptimizer(
#				learning_rate=self.learning_rate_ph,
#			).minimize(self.loss)

	def stack_convolution(self, kernel_size, old_size, new_size, batch_normalization=True):
		weights = tf.Variable(
			# Here we scale down the Xavier initialization due to our residuality.
			0.5 * tf.contrib.layers.xavier_initializer_conv2d()(
				[kernel_size, kernel_size, old_size, new_size],
				dtype=DTYPE,
			),
			name="convkernel",
		)
#		tf.summary.histogram("convkernel", weights)
		self.flow = tf.nn.conv2d(
			self.flow,
			weights,
			strides=[1, 1, 1, 1],
			padding="SAME",
		)
		assert self.flow.dtype == DTYPE
		if batch_normalization:
			self.flow = tf.layers.batch_normalization(
				self.flow,
				training=self.is_training_ph,
#				momentum=0.999,
				center=False,
				scale=False,
#				renorm=True,
#				renorm_momentum=0.999,
			)
			assert self.flow.dtype == DTYPE
		else:
			bias = tf.Variable(tf.constant(0.1, shape=[new_size], dtype=DTYPE), name="bias")
#			tf.summary.histogram("bias", bias)
			self.flow = self.flow + bias

	def stack_nonlinearity(self):
		self.flow = tf.nn.relu(self.flow)
		assert self.flow.dtype == DTYPE

	def stack_block(self):
		initial_value = self.flow
		# Stack the first convolution.
		self.stack_convolution(3, self.FILTERS, self.FILTERS)
		self.stack_nonlinearity()
		# Stack the second convolution.
		self.stack_convolution(3, self.FILTERS, self.FILTERS)
		# Add the skip connection.
		self.flow = self.flow + initial_value
		# Stack on the deferred non-linearity.
		self.stack_nonlinearity()

SAVE_SUFFIX = "model.ckpt"

def save_model(sess, path):
	try:
		os.mkdir(path)
	except OSError:
		pass
	saver = tf.train.Saver()
	saver.save(sess, os.path.join(path, SAVE_SUFFIX))

def load_model(sess, path):
	saver = tf.train.Saver()
	saver.restore(sess, os.path.join(path, SAVE_SUFFIX))

if __name__ == "__main__":
	# Count network parameters.
	net = Network("net/")
	print
	print "Filters:", net.FILTERS
	print "Block count:", net.BLOCK_COUNT

	for var in tf.trainable_variables():
		print var
	print "Parameter count:", net.total_parameters

