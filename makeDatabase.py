<<<<<<< HEAD
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
""" CIFAR 10 dataset (and roomba dataset) constructed like:
<label1 1-byte><red channel of image 1024-byte><green channel of image 1024-byte><blue channel of image 1024-byte><label2 1-byte>...
																												  ^ new image begins here
This is the construction in the .bin file.
In python we create one big list like this:
[label1, red11, red12, ..., red11024, green11, green12, ..., green11024, blue11, blue12, ..., blue11024, label2, ...]
"""
# File converts Roomba images in 32x32 and puts them into a .bin file
# in similar format to CIFAR 10 dataset so roomba dataset can be read
# in tensorflow
def makeDatabase(paths,test,data):

	"""
	Takes a path to a folder containing images, resizes those images
	to 32x32 images and puts them all into a .bin file in the same way
	as the CIFAR-10 dataset.
	:param path: a path to a folder containing images
	:param test: boolean. True if creating test data False if creating eval data
	:param data: int of 0 or 1. 1 if creating roomba eval, 0 if creating grid eval.
	"""
	total = []
	for i in range(len(paths)):
		onlyfiles = [f for f in listdir(paths[i]) if isfile(join(paths[i], f))]
		for n in range(0, len(onlyfiles)):
			im = join(paths[i], onlyfiles[n])
			im = Image.open(im)
			im = im.resize((32,32))
			im = (np.array(im))
			r = im[:, :, 0].flatten()
			g = im[:, :, 1].flatten()
			b = im[:, :, 2].flatten()
			label = [1]
			total += list(label) + list(r) + list(g) + list(b)
	out = np.array(total, np.uint8)
	if test:
		out.tofile(r'roomba_data\roomba_test.bin')
	else:
		if data == 0:
			out.tofile(r'roomba_data\roomba0.bin')
		elif data == 1:
			out.tofile(r'roomba_data\roomba1.bin')
		else:
			print('please enter a valid data num')
			return

def main():
	paths = ['roomba_photos/roomba']
	makeDatabase(paths, False, 1)
if __name__ == "__main__":
=======
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
""" CIFAR 10 dataset (and roomba dataset) constructed like:
<label1 1-byte><red channel of image 1024-byte><green channel of image 1024-byte><blue channel of image 1024-byte><label2 1-byte>...
																												  ^ new image begins here
This is the construction in the .bin file.
In python we create one big list like this:
[label1, red11, red12, ..., red11024, green11, green12, ..., green11024, blue11, blue12, ..., blue11024, label2, ...]
"""
# File converts Roomba images in 32x32 and puts them into a .bin file
# in similar format to CIFAR 10 dataset so roomba dataset can be read
# in tensorflow
def makeDatabase(paths,test,data):

	"""
	Takes a path to a folder containing images, resizes those images
	to 32x32 images and puts them all into a .bin file in the same way
	as the CIFAR-10 dataset.
	:param path: a path to a folder containing images
	:param test: boolean. True if creating test data False if creating eval data
	:param data: int of 0 or 1. 1 if creating roomba eval, 0 if creating grid eval.
	"""
	total = []
	for i in range(len(paths)):
		onlyfiles = [f for f in listdir(paths[i]) if isfile(join(paths[i], f))]
		for n in range(0, len(onlyfiles)):
			im = join(paths[i], onlyfiles[n])
			im = Image.open(im)
			im = im.resize((32,32))
			im = (np.array(im))
			r = im[:, :, 0].flatten()
			g = im[:, :, 1].flatten()
			b = im[:, :, 2].flatten()
			label = [1]
			total += list(label) + list(r) + list(g) + list(b)
	out = np.array(total, np.uint8)
	if test:
		out.tofile(r'roomba_data\roomba_test.bin')
	else:
		if data == 0:
			out.tofile(r'roomba_data\roomba0.bin')
		elif data == 1:
			out.tofile(r'roomba_data\roomba1.bin')
		else:
			print('please enter a valid data num')
			return

def main():
	paths = ['roomba_photos/roomba']
	makeDatabase(paths, False, 1)
if __name__ == "__main__":
>>>>>>> 64c1f092c4c555bd86471ff8f773743a60d239c1
	main()