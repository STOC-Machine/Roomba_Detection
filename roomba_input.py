<<<<<<< HEAD
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size
IMAGE_SIZE = 32

# Global Constants
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2000

def read_roomba(filename_queue):
	"""

	:param filename_queue: A queue of strings with the filenames to read from
	:return:
		An objects representing a single example with fields:
			height: number of rows in result (32)
			width: number of columns in result (32)
			depth: number of color channels in result (3)
			key: a scalar string Tensor describing the filename & record number
				for this example
			label: an int32 Tensor with the label in the range 0..9
			uint8image: a [height, width, depth] uint8 Tensor with the image data
	"""

	class ROOMBARecord(object):
		pass
	result = ROOMBARecord()

	# Dimensions of images in the dataset
	label_bytes = 1
	result.height = 32
	result.width = 32
	result.depth = 3
	image_bytes = result.height * result.width * result.depth
	# Every record consists of a label followed by the image, with a
	# fixed number of bytes for each
	record_bytes = label_bytes + image_bytes

	# Read a record. No header or footer in this format so leave
	# header_bytes and footer_bytes at default of 0.
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

	result.key, value = reader.read(filename_queue)

	# Convert from a string to a vector of uint8 that is record_bytes long.
	record_bytes = tf.decode_raw(value, tf.uint8)

	# The first of the bytes represent the label, which we convert
	# from uint8->int32.
	result.label = tf.cast(
		tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

	# The remaining bytes after the label represent the image, which we reshape
	# from [depth * height * width] to [depth, height, width]
	depth_major = tf.reshape(
		tf.strided_slice(record_bytes, [label_bytes],
						 [label_bytes + image_bytes]),
		[result.depth, result.height, result.width])

	# Convert from [depth, height, width] to [height, width, depth]
	result.uint8image = tf.transpose(depth_major, [1, 2, 0])

	return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
									batch_size, shuffle):
	"""
	Construct a queued batch of images and labels.
	:param image: 3-D Tensor of [height, width, 3] of type.float32
	:param label: 1-D Tensor of type.int32
	:param min_queue_examples: int32, minimum number of samples to retain
		in the queue that provides batches of examples.
	:param batch_size: Number of images per batch
	:param shuffle: boolean indicating whether to use a shuffling queue
	:return: images: 4D tensor of [batch_size, height, width, 3] size.
	:return: labels: 1D tensor of [batch_size] size.
	"""
	# Create a queue that shuffles examples, then read 'batch_size'
	# images + labels from example queue.
	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples+3*batch_size,
			min_after_dequeue=min_queue_examples)

	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples+3*batch_size)

	# Display the training images in the visualizer.
	tf.summary.image('images', images)
	return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size):
	"""
	Construct distorted input for roomba training using the Reader ops.

	:param data_dir: Path the the roomba data directory.
	:param batch_size: Number of images per batch
	:return: images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	:return: labels: 1D tensor of [batch_size] size.
	"""
	filenames = [os.path.join(data_dir, 'roomba_test.bin')]

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a queue that rpoduces the filenames to read.
	filename_queue = tf.train.string_input_producer(filenames)

	# Read examples from files in the filename queue
	read_input = read_roomba(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for training the network. Note the many
	# random distortions applied to the image

	# Randomly crop a [height, width] section of the image.
	distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
	# Randomly flip the image horizontally.
	distorted_image = tf.image.random_flip_left_right(distorted_image)

	distorted_image = tf.image.random_brightness(distorted_image,
												 max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image,
											   lower=0.2, upper=1.8)

	# Subtract off the mean and divide by the variance of the pixels
	float_image = tf.image.per_image_standardization(distorted_image)

	# Set the shapes of tensors.
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has a good mixing of properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*
							 min_fraction_of_examples_in_queue)
	print(min_queue_examples)
	print('Filling the queue with %d roomba images before starting to train.' % min_queue_examples)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label,
										   min_queue_examples, batch_size,
										   shuffle=True)

def inputs(eval_data, data_dir, batch_size):
	"""Construct input for roomba evaluation using the Reader ops.
	:param eval_data: bool, indicating if one should use train or eval data set.
	:param data_dir: Path to the roomba data directory.
	:param batch_size: Number of images per batch.
	:return: images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	:return: labels: 1D tensor of [batch_size] size.
	"""
	if not eval_data:
		filenames = [os.path.join(data_dir, 'roomba%d.bin' % i)
					 for i in range(0,2)]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filenames = [os.path.join(data_dir, 'roomba_test.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a quee that produces the filenames to read.from
	filename_queue = tf.train.string_input_producer(filenames)

	# Read examples from files in the filename queue.
	read_input = read_roomba(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for evaluation
	# Crop the central [height, width] of the image
	resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
														   height, width)

	# Subtract off the mean and divide by the variance of pixels.
	float_image = tf.image.per_image_standardization(resized_image)

	# Set the shapes of tensors.
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has a good mixing of properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch *
							 min_fraction_of_examples_in_queue)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label,
										   min_queue_examples, batch_size,
										   shuffle=False)
=======
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size
IMAGE_SIZE = 32

# Global Constants
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2000

def read_roomba(filename_queue):
	"""

	:param filename_queue: A queue of strings with the filenames to read from
	:return:
		An objects representing a single example with fields:
			height: number of rows in result (32)
			width: number of columns in result (32)
			depth: number of color channels in result (3)
			key: a scalar string Tensor describing the filename & record number
				for this example
			label: an int32 Tensor with the label in the range 0..9
			uint8image: a [height, width, depth] uint8 Tensor with the image data
	"""

	class ROOMBARecord(object):
		pass
	result = ROOMBARecord()

	# Dimensions of images in the dataset
	label_bytes = 1
	result.height = 32
	result.width = 32
	result.depth = 3
	image_bytes = result.height * result.width * result.depth
	# Every record consists of a label followed by the image, with a
	# fixed number of bytes for each
	record_bytes = label_bytes + image_bytes

	# Read a record. No header or footer in this format so leave
	# header_bytes and footer_bytes at default of 0.
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

	result.key, value = reader.read(filename_queue)

	# Convert from a string to a vector of uint8 that is record_bytes long.
	record_bytes = tf.decode_raw(value, tf.uint8)

	# The first of the bytes represent the label, which we convert
	# from uint8->int32.
	result.label = tf.cast(
		tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

	# The remaining bytes after the label represent the image, which we reshape
	# from [depth * height * width] to [depth, height, width]
	depth_major = tf.reshape(
		tf.strided_slice(record_bytes, [label_bytes],
						 [label_bytes + image_bytes]),
		[result.depth, result.height, result.width])

	# Convert from [depth, height, width] to [height, width, depth]
	result.uint8image = tf.transpose(depth_major, [1, 2, 0])

	return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
									batch_size, shuffle):
	"""
	Construct a queued batch of images and labels.
	:param image: 3-D Tensor of [height, width, 3] of type.float32
	:param label: 1-D Tensor of type.int32
	:param min_queue_examples: int32, minimum number of samples to retain
		in the queue that provides batches of examples.
	:param batch_size: Number of images per batch
	:param shuffle: boolean indicating whether to use a shuffling queue
	:return: images: 4D tensor of [batch_size, height, width, 3] size.
	:return: labels: 1D tensor of [batch_size] size.
	"""
	# Create a queue that shuffles examples, then read 'batch_size'
	# images + labels from example queue.
	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples+3*batch_size,
			min_after_dequeue=min_queue_examples)

	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples+3*batch_size)

	# Display the training images in the visualizer.
	tf.summary.image('images', images)
	return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size):
	"""
	Construct distorted input for roomba training using the Reader ops.

	:param data_dir: Path the the roomba data directory.
	:param batch_size: Number of images per batch
	:return: images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	:return: labels: 1D tensor of [batch_size] size.
	"""
	filenames = [os.path.join(data_dir, 'roomba_test.bin')]

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a queue that rpoduces the filenames to read.
	filename_queue = tf.train.string_input_producer(filenames)

	# Read examples from files in the filename queue
	read_input = read_roomba(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for training the network. Note the many
	# random distortions applied to the image

	# Randomly crop a [height, width] section of the image.
	distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
	# Randomly flip the image horizontally.
	distorted_image = tf.image.random_flip_left_right(distorted_image)

	distorted_image = tf.image.random_brightness(distorted_image,
												 max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image,
											   lower=0.2, upper=1.8)

	# Subtract off the mean and divide by the variance of the pixels
	float_image = tf.image.per_image_standardization(distorted_image)

	# Set the shapes of tensors.
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has a good mixing of properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*
							 min_fraction_of_examples_in_queue)
	print(min_queue_examples)
	print('Filling the queue with %d roomba images before starting to train.' % min_queue_examples)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label,
										   min_queue_examples, batch_size,
										   shuffle=True)

def inputs(eval_data, data_dir, batch_size):
	"""Construct input for roomba evaluation using the Reader ops.
	:param eval_data: bool, indicating if one should use train or eval data set.
	:param data_dir: Path to the roomba data directory.
	:param batch_size: Number of images per batch.
	:return: images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	:return: labels: 1D tensor of [batch_size] size.
	"""
	if not eval_data:
		filenames = [os.path.join(data_dir, 'roomba%d.bin' % i)
					 for i in range(0,2)]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filenames = [os.path.join(data_dir, 'roomba_test.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a quee that produces the filenames to read.from
	filename_queue = tf.train.string_input_producer(filenames)

	# Read examples from files in the filename queue.
	read_input = read_roomba(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for evaluation
	# Crop the central [height, width] of the image
	resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
														   height, width)

	# Subtract off the mean and divide by the variance of pixels.
	float_image = tf.image.per_image_standardization(resized_image)

	# Set the shapes of tensors.
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has a good mixing of properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch *
							 min_fraction_of_examples_in_queue)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label,
										   min_queue_examples, batch_size,
										   shuffle=False)
>>>>>>> 64c1f092c4c555bd86471ff8f773743a60d239c1
