from random import shuffle
import glob
import cv2
import numpy as np
import tensorflow as tf
import sys
def list_and_label():
	shuffle_data = True # shuffle the addresses before saving
	roomba_train_path = 'roomba_photos/roomba/*.jpg'

	# read addresses and labels from the 'roomba' folder
	addrs = glob.glob(roomba_train_path)
	labels = [1 for addr in addrs] # 1 = Roomba

	# shuffle data
	if shuffle_data:
		c = list(zip(addrs, labels))
		shuffle(c)
		addrs, labels = zip(*c)

	# Divide teh data in 60% tain, 20% validation, and 20% test
	train_addrs = addrs[0:int(0.6*len(addrs))]
	train_labels = labels[0:int(0.6*len(labels))]

	val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
	val_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]
	test_addrs = addrs[int(0.8*len(addrs)):]
	test_labels = labels[int(0.8*len(labels)):]
	addresses_and_labels = [[train_addrs, val_addrs, test_addrs],
						   [train_labels, val_labels, test_labels]]

	return addresses_and_labels

def load_image(addr):
	# read an image and resize to (224, 224)
	# cv2 load images as BGR, convert it to RGB
	img = cv2.imread(addr)
	img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.astype(np.float32)
	return img

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_train(train_data):
	train_filename ='roomba_data/train.tfrecords' # address to save the TFRecords file

	train_addrs = train_data[0]
	train_labels = train_data[1]

	# open the TFRecords file
	writer = tf.python_io.TFRecordWriter(train_filename)

	for i in range(len(train_addrs)):

		# load the image
		img = load_image(train_addrs[i])

		label = train_labels[i]

		# create a feature
		feature = {'train/label':_int64_feature(label),
			   	'train/image':_bytes_feature(tf.compat.as_bytes(img.tostring()))}

		# create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())

	writer.close()
	sys.stdout.flush()

def write_val(val_data):
	val_addrs = val_data[0]
	val_labels = val_data[1]

	val_filename = 'roomba_data/val.tfrecords' # address to save the TFRecords file
	writer = tf.python_io.TFRecordWriter(val_filename)
	for i in range(len(val_addrs)):
		# load the image
		img = load_image(val_addrs[i])
		label = val_labels[i]
		# create a feature
		feature = {'val/label': _int64_feature(label),
				   'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

		# Create an example protocol buffer
		example = tf.train.Example(features = tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())

	writer.close()
	sys.stdout.flush()

def write_test(test_data):
	test_addrs = test_data[0]
	test_labels = test_data[1]

	test_filename = 'roomba_data/test.tfrecords' # address to save the TFRecords file
	writer = tf.python_io.TFRecordWriter(test_filename)

	for i in range(len(test_addrs)):
		# load the image
		img = load_image(test_addrs[i])

		label = test_labels[i]

		# create a feature
		feature = {'test/label': _int64_feature(label),
				   'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write to the file
		writer.write(example.SerializeToString())

	writer.close()
	sys.stdout.flush()

def main():
	data = list_and_label()
	train_data = [data[0][0], data[1][0]]
	val_data = [data[0][1], data[1][1]]
	test_data = [data[0][2], data[1][2]]
	write_train(train_data)
	write_val(val_data)
	write_test(test_data)

if __name__ == "__main__":
	main()





