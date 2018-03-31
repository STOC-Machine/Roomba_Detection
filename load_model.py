import tensorflow as tf
import cv2

cam = cv2.VideoCapture(0)
with tf.Session() as sess:
	model_saver = tf.train.import_meta_graph("cnn_files\model.ckpt-8992.meta")
	model_saver.restore(sess, "cnn_files\model.ckpt-10000")

