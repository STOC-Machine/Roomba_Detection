from datetime import datetime
import math
import time
import cv2
import numpy as np
import tensorflow as tf
import roomba

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', 'cnn_files',
                           """Directory where to read model checkpoints.""")


def  evaluate_images(images):
	logit = roomba.inferences(images)
	load_trained_model(logit)

def load_trained_model(logits):
	ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	input_ckpt = ckpt.model_checkpoint_path
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(input_ckpt + '.meta', clear_devices=True)
		sess.run(tf.initialize_all_variables())
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		saver.restore(sess, ckpt.model_checkpoint_path)

		predict = tf.argmax(logits,1)
		print(predict.eval(), '\n')

def img_read(cam):
	_, frame = cam.read()
	frame = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_CUBIC)

	image = tf.Variable(frame)
	image = tf.cast(image, tf.float32)
	image = tf.expand_dims(image, 0, name="input")
	return image
cam = cv2.VideoCapture(0)
while True:
	images = img_read(cam)
	evaluate_images(images)