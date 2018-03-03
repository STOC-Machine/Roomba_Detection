"""Converts checkpoint variables into Const ops in standalone GraphDef file."""

import os
import tensorflow as tf

def freeze_graph(model_dir, output_node_names):
	"""Extract the sub graph defined by the output nodes and convert
	all its variables into constant

	"""

	# Retrieve checkpoint fullpath
	ckpt = tf.train.get_checkpoint_state(model_dir)
	input_ckpt = ckpt.model_checkpoint_path

	# precise the file fullname of freezed graph
	output_graph = "cnn_files/frozen_model.pb"

	# Clear devices to allow TF to control which device loads operations
	clear_devices = True

	# Start session using a temp Graph
	with tf.Session(graph=tf.Graph()) as sess:
		# import the meta graph in the current default Graph
		saver = tf.train.import_meta_graph(input_ckpt + '.meta', clear_devices=clear_devices)

		# Restore weights
		saver.restore(sess, input_ckpt)

		# Use built-in TF helper to export variables to constants
		output_graph_def = tf.graph_util.convert_variables_to_constants(
			sess, # The session is used to retrieve the weights
			tf.get_default_graph().as_graph_def(), # graph_def used to retrieve nodes
			output_node_names.split(",") # output node names are used to select usefull nodes
		)

		# Serialize and dump the output graph to the filesystem
		with tf.gfile.GFile(output_graph, 'wb') as f:
			f.write(output_graph_def.SerializeToString())

		print("%d ops in the final graph." %len(output_graph_def.node))

	return output_graph_def

def main():

	freeze_graph('cnn_files', 'softmax_linear/softmax_linear,images')
if __name__ == "__main__":
	main()