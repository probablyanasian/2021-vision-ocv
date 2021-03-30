# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from networktables import NetworkTables
# from sympy import Interval, Union # not performant enough with multiple interval joins.
import numpy as np
import argparse
import threading
import cv2
import imutils
import time

# Normal image, Filter image
DEBUG = {
	'show_img': True,
	'show_filter': True,
	'show_centroid': True
	}
REQ_CLOSEST = False
CONNECT_TO_SERVER = False
CENTER_BAND = 100
HORIZONTAL_OFFSET = 100

def connect():
    cond = threading.Condition()
    notified = [False]

    def connectionListener(connected, info):
        print(info, '; Connected=%s' % connected)
        with cond:
            notified[0] = True
            cond.notify()

    NetworkTables.initialize(server='roborio-2643-frc.local')
    NetworkTables.addConnectionListener(
        connectionListener, immediateNotify=True)

    with cond:
        print("Waiting")
        if not notified[0]:
            cond.wait()

    return NetworkTables.getTable('gs-vision-table-marker')


if CONNECT_TO_SERVER:
    table = connect()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())


# TODO: make cyan by image color inversion... perhaps
# Invert and convert to HSV

# img_hsv=cv2.cvtColor(255-img, cv2.COLOR_BGR2HSV) 

# # Mask all red pixels (cyan in inverted image)
# lo = np.uint8([80,30,0]) 
# hi = np.uint8([95,255,255])  

# mask = cv2.inRange(img_hsv,lo,hi)

redLower = (170, 0, 0)    # TODO: get vals. 
redUpper = (179, 255, 255) # TODO: get vals. 

blueLower = (100, 80, 0)     # TODO: get vals. higher saturation I think. b/c MPR light blue.
blueUpper = (110, 255, 255)   # TODO: get vals. 

minArea = 50 # 10 TODO: tune.
pts = deque(maxlen=args["buffer"])

vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vs.set(cv2.CAP_PROP_FPS, 30)

time.sleep(2.0)

img_x_size = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
img_y_size = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
img_center = (img_x_size//2, img_y_size//2)

# allow the camera or video file to warm up

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1]

	# frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color, then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	red_mask = cv2.inRange(hsv, redLower, redUpper)
	if DEBUG['show_filter']:
		cv2.imshow("red_filter", red_mask)
	red_mask = cv2.erode(red_mask, None, iterations=2)
	red_mask = cv2.dilate(red_mask, None, iterations=2)
	
	red_cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	red_cnts = imutils.grab_contours(red_cnts)

	blue_mask = cv2.inRange(hsv, blueLower, blueUpper)
	if DEBUG['show_filter']:
		cv2.imshow("blue_filter", blue_mask)
	blue_mask = cv2.erode(blue_mask, None, iterations=2)
	blue_mask = cv2.dilate(blue_mask, None, iterations=2)

	# find contours in the mask
	blue_cnts = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	blue_cnts = imutils.grab_contours(blue_cnts)
	
	center = None

	# only proceed if at least one contour was found
	valid_red_cnts = []
	if len(red_cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		for c in red_cnts:
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if M["m00"] > minArea:
				rect = cv2.minAreaRect(c)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				cv2.drawContours(frame,[box],0,(0,0,255),2)
				valid_red_cnts.append({
					'contour': c,
					'left_edge': min(box[0][0], box[3][0]),
					'right_edge': max(box[1][0], box[2][0]),
					'top_edge': min(box[0][1], box[1][1]),
					'bottom_edge': min(box[2][1], box[3][1])
				})
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
		
		if REQ_CLOSEST:
			#track the largest
			c = max(red_cnts, key=cv2.contourArea)
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
			print(box)
			frame = cv2.line(frame,(min(box[0][0], box[3][0]), 0),(min(box[0][0], box[3][0]), img_y_size),(252, 3, 119),3)
			frame = cv2.line(frame,(max(box[1][0], box[2][0]), 0),(max(box[1][0], box[2][0]), img_y_size),(252, 3, 119),3)

			box = np.int0(box)
			cv2.drawContours(frame,[box],0,(0,0,255),2)

			M = cv2.moments(c)
			if M["m00"] > minArea:
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			else:
				center = None
	else:
		center = None

	if CONNECT_TO_SERVER:
		if center is None:
			table.putBoolean('has_target', False)
			table.putBoolean('near', False)
		else:
			table.putBoolean('has_target', True)
			if center[0] < (img_center[0] - CENTER_BAND):
				table.putNumber('left_exceeded', ((img_center[0] - CENTER_BAND) - center[0]))
			else:
				table.putNumber('left_exceeded', 0)

			if center[0] > (img_center[0] + CENTER_BAND):
				table.putNumber('right_exceeded', (center[0] - (img_center[0] + CENTER_BAND)))
			else:
				table.putNumber('right_exceeded', 0)
			
			if center[1] < (img_center[1] + HORIZONTAL_OFFSET):
				table.putBoolean('near', True)
			else:
				table.putBoolean('near', False)


	# update the points queue
	if DEBUG['show_img'] and REQ_CLOSEST:
		pts.appendleft(center)

		# loop over the set of tracked points
		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue

			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	if DEBUG['show_img']:
		cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()