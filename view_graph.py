import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
	model_filename ='frozen_model.pb'
	with gfile.FastGFile(model_filename, 'rb') as f:
		graph_def = tf.GraphDef()
		softmax_tensor = sess.graph.get_tensor_by_name('softmax_linear/softmax_linear:0')
		graph_def.ParseFromString(f.read())
		g_in = tf.import_graph_def(graph_def)
LOGDIR=''
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()