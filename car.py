import serial
import cv2
import numpy as np
import glob
import imutils
from mss import mss
from PIL import Image
import time

def get_mask_obstacle(X, Y, width, up, x1=None, y1=None, x2=None, y2=None, lane_shift=0, angle_shift=0):
	width = int(X*width)
	up = int(Y*up)
	down = int(Y-1)
	height = down - up
	figure = np.zeros((Y, X), dtype = 'uint8')
	for x in range(X):
		for y in range(up, down+1):
			if abs(x-X/2 - ((y-up)/height*3 + 1)*lane_shift - (1-y/Y)*angle_shift) <= ((y-up)/height*3 + 1)*width:
				figure[y, x] = 255
	return figure

def main():
	mon = {'top': 50, 'left': 0, 'width': 640, 'height': 480}
	sct = mss()
	forward = b"\x01"
	turn_left = b"\x02"
	turn_right = b"\x03"
	move_to_left_lane = b"\x04"
	move_to_right_lane = b"\x05"
	stop = b"\x06"

	ser = serial.Serial(
		port='/dev/cu.usbmodem14101',
		baudrate=115200,
		parity=serial.PARITY_ODD,
		stopbits=serial.STOPBITS_TWO,
		bytesize=serial.SEVENBITS
	)
	ser.isOpen()
	# album = [cv2.imread(file) for file in glob.glob("./right/*.jpg")]
	sct.get_pixels(mon)
	img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
	img = np.array(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
	img = img[int(img.shape[0]*1/5):img.shape[0], :]
	X, Y = img.shape[1], img.shape[0]
	right_lane_mask = get_mask_obstacle(X, Y, width = 1/20, up = 3/10, lane_shift = 130, angle_shift = 0)
	left_lane_mask = get_mask_obstacle(X, Y, width = 1/20, up = 3/10, lane_shift = -130, angle_shift = 0)
	central_mask = get_mask_obstacle(X, Y, width = 1/12, up = 3/10, lane_shift = 0, angle_shift = 0)
	turn_right_mask = get_mask_obstacle(X, Y, width = 1/12, up = 7/10, lane_shift = 0, angle_shift = 800) # get_mask_right(X, Y)
	turn_left_mask = get_mask_obstacle(X, Y, width = 1/12, up = 7/10, lane_shift = 0, angle_shift = -800) # get_mask_left(X, Y)
	# for orig_img in album:
	# # album.sort(key = getName)
	while True:
		sct.get_pixels(mon)
		orig_img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
		orig_img = np.array(orig_img)
		orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
		M = cv2.getRotationMatrix2D((int(orig_img.shape[1]/2), int(orig_img.shape[0])/2), 180, 1.0)
		orig_img = cv2.warpAffine(orig_img, M, (int(orig_img.shape[1]), int(orig_img.shape[0])))
		orig_img = cv2.resize(orig_img, (int(orig_img.shape[1]/2), int(orig_img.shape[0]/2)))
		# orig_img = cv2.resize(orig_img, (int(orig_img.shape[1]/10), int(orig_img.shape[0]/10)))
		orig_img = orig_img[int(orig_img.shape[0]*1/5):orig_img.shape[0], :]
		img = orig_img.copy()
		# figure = get_mask_left(img.shape[1], img.shape[0])
		# for orig_img in album:
		ret, threshold = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
		kernel1 = np.ones((3, 3), np.uint8)
		# kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
		erosion = cv2.erode(threshold, kernel1, iterations = 1)
		dilation = cv2.dilate(erosion, kernel1, iterations = 4)
		img = dilation.copy()
		cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		mask = np.ones(img.shape[:2], dtype="uint8") * 255

		for c in cnts:
			area = cv2.contourArea(c)
			if area < 3000:
				cv2.drawContours(mask, [c], -1, 0, -1)
				M = cv2.moments(c)
				if M["m00"] != 0:
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
					cv2.circle(img, (cX, cY), int((area**0.5)**0.5), (0), -1)
		img = cv2.bitwise_and(img, img, mask=mask)
		right_view = img + turn_right_mask
		left_view = img + turn_left_mask
		central_view = img + central_mask
		right_lane_view = img + right_lane_mask
		left_lane_view = img + left_lane_mask
		right_hit = cv2.bitwise_and(img, img, mask=turn_right_mask)
		left_hit = cv2.bitwise_and(img, img, mask=turn_left_mask)
		central_hit = cv2.bitwise_and(img, img, mask=central_mask)
		right_lane_hit = cv2.bitwise_and(img, img, mask=right_lane_mask)
		left_lane_hit = cv2.bitwise_and(img, img, mask=left_lane_mask)
		if cv2.countNonZero(central_hit) < 50:
			command = forward
		elif cv2.countNonZero(right_hit) < 50 and cv2.countNonZero(left_lane_hit) > 50:
			command = turn_right
		elif cv2.countNonZero(left_hit) < 50 and cv2.countNonZero(right_lane_hit) > 50:
			command = turn_left
		elif cv2.countNonZero(left_lane_hit) < 50 and cv2.countNonZero(right_lane_hit) < 50:
			if cv2.countNonZero(right_hit) < cv2.countNonZero(left_hit):
				command = turn_right
			else:
				command = turn_left
		else:
			command = stop
		ser.write(command)
		print(command)
		time.sleep(0.35)
		img = np.vstack((np.hstack((orig_img, threshold, erosion)), np.hstack((dilation, mask, img)), np.hstack((central_mask, central_view, central_hit)), np.hstack((left_lane_mask, left_lane_view, left_lane_hit)), np.hstack((turn_left_mask, left_view, left_hit)), np.hstack((turn_right_mask, right_view, right_hit)), np.hstack((right_lane_mask, right_lane_view, right_lane_hit))))
		img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
		window = 'image'
		cv2.namedWindow(window)
		cv2.moveWindow(window, 650, 50)
		cv2.imshow(window, img)
		cv2.waitKey(50)

if __name__ == '__main__':
	main()