<<<<<<< HEAD
import cv2
import os

def image_grabber(filename):
	feed = cv2.VideoCapture(1)
	num= 500
	capture_frame = 1
	frame_idx = 0
	while num < 1000:
		_, frame = feed.read()
		if frame_idx % capture_frame== 0:
			file = os.path.join(filename, 'img%d.JPEG' % num)
			cv2.imwrite(filename=file, img=frame,
						params= [int(cv2.IMWRITE_JPEG_QUALITY), 100])
			num += 1
		frame_idx += 1
def main():
	filename = r'roomba_photos\roomba'
	image_grabber(filename=filename)

if __name__ == "__main__":
=======
import cv2
import os

def image_grabber(filename):
	feed = cv2.VideoCapture(1)
	num= 500
	capture_frame = 1
	frame_idx = 0
	while num < 1000:
		_, frame = feed.read()
		if frame_idx % capture_frame== 0:
			file = os.path.join(filename, 'img%d.JPEG' % num)
			cv2.imwrite(filename=file, img=frame,
						params= [int(cv2.IMWRITE_JPEG_QUALITY), 100])
			num += 1
		frame_idx += 1
def main():
	filename = r'roomba_photos\roomba'
	image_grabber(filename=filename)

if __name__ == "__main__":
>>>>>>> 64c1f092c4c555bd86471ff8f773743a60d239c1
	main()