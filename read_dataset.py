<<<<<<< HEAD
import tensorflow as tf
import numpy as np
import random

def read_dataset(type):
	"""
	reads data from tfrecords
	:param type: int 0<x<3; 0 = training dataset, 1 = validation dataset,
		2 = test dataset
	:return: img: A np.array of 3 channel images
			 lbl: A np.array of length img with labels for each image
	"""
	if type == 0:
		data_path = 'roomba_data/train.tfrecords'  # address to save the hdf5 file
		image_name = 'train/image'
		label_name = 'train/label'

	elif type == 1:
		data_path = 'roomba_data/val.tfrecords'
		image_name = 'val/image'
		label_name = 'val/label'

	else:
		data_path = 'roomba_data/test.tfrecords'
		image_name = 'test/image'
		label_name = 'test/label'


	with tf.Session() as sess:
		feature = {image_name: tf.FixedLenFeature([], tf.string),
				   label_name: tf.FixedLenFeature([], tf.int64)}
		
		# Create a list of filenames and pass it to a queue
		filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
		# Define a reader and read the next record
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		# Decode the record read by the reader
		features = tf.parse_single_example(serialized_example, features=feature)
		# Convert the image data from string back to the numbers
		image = tf.decode_raw(features[image_name], tf.float32)

		# Cast label data into int32
		label = tf.cast(features[label_name], tf.int32)
		# Reshape image data into the original shape
		image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
		# Any preprocessing here ...
		# Creates batches by randomly shuffling tensors
		images, labels = tf.train.shuffle_batch([image, label], batch_size=BATCH_SIZE, capacity=30, num_threads=1,
												min_after_dequeue=10)

		# Initailize all global and local variables
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		# Create a coordinator and run all QueueRunner objects
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		#for batch_index in range(5):
		img, lbl = sess.run([images, labels])
		img = np.array(img)
		lbl = np.array(lbl)
		# Stop the threads
		coord.request_stop()

		# Wait for threads to stop
		coord.join(threads)
		sess.close()
		return img, lbl
=======
import tensorflow as tf
import numpy as np
import random

def read_dataset(type):
	"""
	reads data from tfrecords
	:param type: int 0<x<3; 0 = training dataset, 1 = validation dataset,
		2 = test dataset
	:return: img: A np.array of 3 channel images
			 lbl: A np.array of length img with labels for each image
	"""
	if type == 0:
		data_path = 'roomba_data/train.tfrecords'  # address to save the hdf5 file
		image_name = 'train/image'
		label_name = 'train/label'

	elif type == 1:
		data_path = 'roomba_data/val.tfrecords'
		image_name = 'val/image'
		label_name = 'val/label'

	else:
		data_path = 'roomba_data/test.tfrecords'
		image_name = 'test/image'
		label_name = 'test/label'


	with tf.Session() as sess:
		feature = {image_name: tf.FixedLenFeature([], tf.string),
				   label_name: tf.FixedLenFeature([], tf.int64)}
		
		# Create a list of filenames and pass it to a queue
		filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
		# Define a reader and read the next record
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		# Decode the record read by the reader
		features = tf.parse_single_example(serialized_example, features=feature)
		# Convert the image data from string back to the numbers
		image = tf.decode_raw(features[image_name], tf.float32)

		# Cast label data into int32
		label = tf.cast(features[label_name], tf.int32)
		# Reshape image data into the original shape
		image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
		# Any preprocessing here ...
		# Creates batches by randomly shuffling tensors
		images, labels = tf.train.shuffle_batch([image, label], batch_size=BATCH_SIZE, capacity=30, num_threads=1,
												min_after_dequeue=10)

		# Initailize all global and local variables
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		# Create a coordinator and run all QueueRunner objects
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		#for batch_index in range(5):
		img, lbl = sess.run([images, labels])
		img = np.array(img)
		lbl = np.array(lbl)
		# Stop the threads
		coord.request_stop()

		# Wait for threads to stop
		coord.join(threads)
		sess.close()
		return img, lbl
>>>>>>> 64c1f092c4c555bd86471ff8f773743a60d239c1
