import cv2
import numpy as np
import tensorflow as tf
import roomba

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', r'cnn_files',
						   """Directory where to read model checkpoints.""")
MOVING_AVERAGE_DECAY = 0.9999    # The deacy to use for the moving average.
height = 32
width = 32
depth = 3

def find_roomba():
	x = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder('float')
	#saver = tf.train.Saver()
	feed = cv2.VideoCapture(0)
	graph = tf.Graph()
	with graph.as_default():
		while True:
			# Initialize image as list that neural net can read
			_, frame = feed.read()
			frame = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_CUBIC)
			frame = np.array(frame)
			r = frame[:, :, 0].flatten()
			g = frame[:, :, 1].flatten()
			b = frame[:, :, 2].flatten()
			image = np.array([list(r) + list(g) + list(b)], np.uint8)
			image = tf.Variable(image)
			depth_major = tf.reshape(image, [depth, height, width])

			# Convert from [depth, height, width] to [height, width, depth]
			uint8image = tf.transpose(depth_major, [1, 2, 0])
			image = tf.cast(uint8image, tf.float32)
			# create a batch of 1
			image = tf.reshape(image, [1,32,32,3])
			logits = roomba.inferences(image)
			print(logits)
			with tf.Session() as sess:
				output = sess.run(logits, feed_dict={x : image, keep_prob : 0.8})
				print(output)
def main():
	find_roomba()

if __name__ == "__main__":
	main()



