import cv2
import numpy as np

def find_roomba():
	feed = cv2.VideoCapture(0)
	while True:
		_, frame = feed.read()
		frame = cv2.resize(frame, (32,32), interpolation=cv2.INTER_CUBIC)
		frame = np.array(frame)
		r = frame[:,:,0].flatten()
		g = frame[:,:,1].flatten()
		b = frame[:,:,2].flatten()
		image = np.array([list(r) + list(g) + list(b)], np.uint8)
		"""
		Insert neural network to evaluate image here
		"""



