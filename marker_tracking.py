# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from networktables import NetworkTables
import numpy as np
import argparse
import threading
import cv2
import imutils
import time

# Normal image, Filter image
DEBUG = [True, True]
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

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# TODO: make cyan by image color inversion... perhaps
# Invert and convert to HSV

# img_hsv=cv2.cvtColor(255-img, cv2.COLOR_BGR2HSV) 

# # Mask all red pixels (cyan in inverted image)
# lo = np.uint8([80,30,0]) 
# hi = np.uint8([95,255,255])  

# mask = cv2.inRange(img_hsv,lo,hi)

redLower = (16, 0, 64)    # TODO: get vals. 
redUpper = (24, 255, 255) # TODO: get vals. 

blueLower = (0, 0, 0)     # TODO: get vals.
blueUpper = (179, 0, 0)   # TODO: get vals. 

minRadius = 15 # 10
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam

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

	# resize the frame, blur it, and convert it to the HSV
	# color space
	# frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, yellowLower, yellowUpper)
	if DEBUG[1]:
		cv2.imshow("filter", mask)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		for c in cnts:
			# c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if radius > minRadius:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
		
		#track the largest
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		if radius > minRadius:
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
	if DEBUG[0]:
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