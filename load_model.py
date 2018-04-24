import cv2
import numpy as np
import tensorflow as tf
import roomba

cam = cv2.VideoCapture(0)

while True:
	_, frame = cam.read()
	image = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_CUBIC)
	image = np.expand_dims(image, axis=0)
	image = tf.cast(image, tf.float32)
	softmax = roomba.inferences(image)
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state('cnn_files')
	with tf.Session() as sess:
		saver.restore(sess, ckpt.model_checkpoint_path)
		softmaxval = sess.run(softmax)
		print(softmaxval)
