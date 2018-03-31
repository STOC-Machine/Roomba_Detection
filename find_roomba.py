import cv2
import tensorflow as tf
import numpy as np

sess = tf.Session('', tf.Graph())
cam = cv2.VideoCapture(0)
with sess.graph.as_default():
	# Read meta graph and checkpoint to restore session
	saver = tf.train.import_meta_graph("cnn_files\model.ckpt-8872.meta")
	saver.restore(sess,"cnn_files\model.ckpt-8872")
	_, frame = cam.read()
	image = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_CUBIC)
	image = np.expand_dims(image, axis=0)
	print(image.shape)
	# Start the queue runners. If they are not started the program will hang
	coord = tf.train.Coordinator()
	threads =[]
	for qr in sess.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
		threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
										 start=True))

	# In the graph created above, feed "is_training" and "imgs" placeholders
	# Feeding them will disconnect the path from queue runners to the graph
	# and enable a path from the placeholder instead. The "img" placeholder will be
	# fed with the image that was read above.
	logits = sess.run('softmax_linear/softmax_linear:0',
					  feed_dict={'is_training:0': False, 'imgs:0':image})

	#Print classification results
	print(logits)

