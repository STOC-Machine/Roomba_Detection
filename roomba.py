"""Builds roomba network
Function summary:

	# Compute input images and labels for training.
	# Run inputs() to run evaluations
	inputs, labels = distorted_inputs()

	# Compute the total loss of prediction with respect to the labels.
	loss = loss(predictions, labels)

	# Create a graph to run one step of training with respect to the loss.
	train_op = train(loss, global_step)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re


import tensorflow as tf

import roomba_input

FLAGS = tf.app.flags.FLAGS

# Basic model params
tf.app.flags.DEFINE_integer('batch_size', 128,
							"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', r'roomba_data',
						   """Path to the roomba data directory""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
							"""Train the model using fp16.""")

# Global constants describing the roomba data set.
IMAGE_SIZE = roomba_input.IMAGE_SIZE
NUM_CLASSES = roomba_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = roomba_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = roomba_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999    # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1 # Learning rate decay factor
INITIAL_LEARNING_RATE = 0.1      # Initial learning rate

TOWER_NAME = 'tower'

def _activation_summary(x):
	"""Helper to create summaries for activations
	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.

	:param x: Tensor
	:return: Nothing
	"""
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations',x)
	tf.summary.scalar(tensor_name + '/sparcity',
					  tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory
	:param name: name of the variable
	:param shape: list of ints
	:param initializer: initializer for Variable
	:return: var: Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.
	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.
	:param name: name of the variable
	:param shape: list of ints
	:param stddev: standard deviation of a truncated Gaussian
	:param wd: add L2Loss weight decay multiplied by this float. If None, weight
				decay is not added for this Variable
	:return: var: Variable Tensor
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(name, shape,
						   tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def distorted_inputs():
	"""Construct distorted input for roomba training using the Reader ops.

	:return: images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	:return: labels: 1D tensor of [batch_size] size.

	Raises:
		ValueError: If no data_dir
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	images, labels = roomba_input.distorted_inputs(data_dir=FLAGS.data_dir,
												   batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels

def inputs(eval_data):
	"""Construct input for roomba evaluation using the Reader ops.

	:param eval_data: bool, indicating if one should use the train or eval data set.

	:return: images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	:return: labels: 1D tensor of [batch_size] size.

	Raises:
		ValueError: If no data_dir
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	images, labels = roomba_input.inputs(eval_data=eval_data,
										 data_dir=FLAGS.data_dir,
										 batch_size=FLAGS.batch_size)

	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels

def inferences(images):
	""" Build the roomba model

	:param images: Images returned from distorted_inputs() or inputs()
	:param batch_size: Number of images for batch
	:return: Logits
	"""
	# conv1
	with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE) as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5,5,3,64],
											 stddev=5e-2,
											 wd=None)
		conv =tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv1)

    # conv2
	with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5,5,64,64],
											 stddev=5e-2,
											 wd=None)
		conv = tf.nn.conv2d(conv1,kernel, [1,1,1,1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv2)

	# pool2
	pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1],
						   strides=[1,4,4,1], padding='SAME', name='pool2')


	# local3
	with tf.variable_scope('local3', reuse=tf.AUTO_REUSE) as scope:
		# Move everything into depth so we can perform a single matrix multiply
		# reshapes to proper batch_size: 1 if implementing FLAGS.BATCH_SIZE if testing or evaluating.
		reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
		dim = reshape.get_shape()[1].value

		weights = _variable_with_weight_decay('weights', shape=[dim, 384],
											  stddev=0.04, wd =0.004)
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
		_activation_summary(local3)


	# local4
	with tf.variable_scope('local4',reuse=tf.AUTO_REUSE) as scope:
		weights = _variable_with_weight_decay('weights', shape=[384, 192],
											  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		_activation_summary(local4)


	# linear layer(WX +b),
	# We don't apply softmax here b/c
	# tf.nn.sparse_softmax_cross_entropy_with_logits accepts hte unscaled logits
	# and performs the softmax internally for efficiency.
	with tf.variable_scope('softmax_linear',reuse=tf.AUTO_REUSE) as scope:
		weights = _variable_with_weight_decay('weights', [192,2],
											  stddev=1/192.0, wd=0.0)
		biases = _variable_on_cpu('biases', [NUM_CLASSES],
								  tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)
	return softmax_linear

def loss(logits, labels):
	""" Add L2Loss to all the training variables.
	Add summary for "Loss" and "Loss/avg"
	:param logits: Logits from inference()
	:param labels: Labels from distorted_inputs or inputs(). 1-D tensor
					of shape [batch_size]
	:return: Loss tensor of type float
	"""
	# Calculate the average cross entropy loss across the batch.
	labels = tf.cast(labels, tf.int64)

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name ='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
	"""Add summaries for losses in roomba model.

	Generates moving average for all losses and associated summaries for
	visualizing the performace of the network.

	:param total_loss: Total loss from loss()
	:return: loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses =tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the average version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op

def train(total_loss, global_step):
	"""Train the roomba model.

	Createa an optimzer and apply to all trainable variables. Add moving
	average for all training variables

	:param total_loss: Total loss form loss()
	:param global_step: Integer Variable counting the number of training steps
			processed.
	:return: train_op: op for training
	"""
	#Variables that affect leanring rate.
	num_batch_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batch_per_epoch*NUM_EPOCHS_PER_DECAY)

	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
									global_step,
									decay_steps,
									LEARNING_RATE_DECAY_FACTOR,
									staircase=True)
	tf.summary.scalar('learning_rate', lr)

	# Generate moving averages of all losses and associated summaries
	loss_averages_op = _add_loss_summaries(total_loss)

	# Compute gradients
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	# Apply gradients
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	# Add histograms for gradients
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')
	return train_op

