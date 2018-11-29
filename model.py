#!/usr/bin/python

import numpy as np
import tensorflow as tf

#DTYPE = tf.float16
DTYPE = tf.float32

class ResNet:
	INPUT_FEATURE_COUNT = 2
	FILTERS = 32
	CONV_SIZE = 3
	BLOCK_COUNT = 8
	OUTPUT_SOFTMAX_COUNT = 16 * 16

	def __init__(self, scope_name):
		with tf.variable_scope(scope_name):
			self.build_graph()
			with tf.variable_scope("training"):
				self.build_training()

	def build_graph(self):
		self.input_ph = tf.placeholder(
			dtype=DTYPE,
			shape=[None, 19, 19, self.INPUT_FEATURE_COUNT],
			name="input_ph",
		)
		self.desired_policy_ph = tf.placeholder(
			dtype=DTYPE,
			shape=[None, 19, 19],
			name="desired_policy_ph",
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
			self.stack_convolution(3, self.INPUT_FEATURE_COUNT, self.FILTERS)
			self.stack_nonlinearity()
		# Stack some number of residual blocks.
		for i in xrange(self.BLOCK_COUNT):
			with tf.variable_scope("block%i" % i):
				self.stack_block()
		# Stack a final batch-unnormalized 1x1 convolution.
		self.stack_convolution(1, self.FILTERS, 1, batch_normalization=False)

	def build_training(self):
		# Construct the training components.
		self.flattened = tf.reshape(self.flow, [-1, self.OUTPUT_SOFTMAX_COUNT])
		self.flattened_desired_output = tf.reshape(self.desired_policy_ph, [-1, self.OUTPUT_SOFTMAX_COUNT])
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
			labels=self.flattened_desired_output,
			logits=self.flattened,
		))

		tf.summary.scalar("cross_entropy", self.cross_entropy)

		regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
		reg_variables = tf.trainable_variables()
		self.regularization_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
		self.loss = self.cross_entropy + self.regularization_term

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

if __name__ == "__main__":
	# Count network parameters.
	net = ResNet("net/")
	print
	print "Filters:", net.FILTERS
	print "Block count:", net.BLOCK_COUNT

	for var in tf.trainable_variables():
		print var
	parameter_count = sum(np.product(var.shape) for var in tf.trainable_variables())
	print "Parameter count:", parameter_count

