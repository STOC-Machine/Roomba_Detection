import os
import timeit
import numpy as np
import cv2
import tensorflow as tf
import roomba


def grab_frame(cam):
	_, frame = cam.read()
	return frame if _ else None

def setup():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	start_time = timeit.default_timer()

	# Takes 2-5 sec to run
	# Unpersists graph from file
	with tf.gfile.FastGFile('cnn_files/frozen_model.pb', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')

	print('Took {} seconds to unpersist the graph'.format(timeit.default_timer()-start_time))


def find_roomba():
	cam = cv2.VideoCapture(0)
	setup()
	while True:
		frame = grab_frame(cam)
		# Get image from video feed
		cv2.imshow('Main', frame)

		frame = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_CUBIC)

		image = tf.Variable(frame)
		image = tf.cast(image, tf.float32)
		image = tf.expand_dims(image, 0, name = "input")
		print(image)
		with tf.Session() as sess:
			print("***************** Session Start *****************")
			start_time = timeit.default_timer()
			# Feed the image_data as input to the graph and get first prediction

			tensor_names = [t.name for op in tf.get_default_graph().get_operations() for t in op.values()]
			print(tensor_names)
			print(sess.graph.get_operations())
			batch = sess.graph.get_tensor_by_name('shuffle_batch/n:0')
			print(tf.shape(batch))
			softmax_tensor = sess.graph.get_tensor_by_name('softmax_linear/softmax_linear:0')

			print('Tensor', softmax_tensor)

			print('Took {} seconds to feed data to graph'.format(timeit.default_timer()-start_time))

			# convert tensor image to array
			#sess.run(tf.global_variables_initializer())
			#image = sess.run(image)

			start_time = timeit.default_timer()

			# Takes 2-5 seconds
			predictions = sess.run(softmax_tensor,{'softmax_tensor':'input:0'})

			print('Took {} seconds to perform prediction'.format(timeit.default_timer()-start_time))

			start_time = timeit.default_timer()

			# Sort to show labels of first prediction in order of confidence
			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

			print('Took {} seconds to sort predictions'.format(timeit.default_timer()-start_time))

			for node_id in top_k:
				score = predictions[0][node_id]
				print(score)

			print('***************** Session Ended *****************')

			if cv2.waitKey(1) & 0xFF == ord('q'):
				sess.close()
				break
	cam.release()
	cv2.destroyAllWindows()


def main():
	find_roomba()
if __name__ == '__main__':
	main()