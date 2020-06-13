# USAGE
# python gather.py -t real
# python gather.py -t fake

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, required=True,
	help="fake or real")
ap.add_argument("-i", "--input", type=str, default="videos",
	help="path to input video")
ap.add_argument("-o", "--output", type=str, default="dataset",
	help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, default="face_detector",
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=8,
	help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
inp = os.path.sep.join([args["input"], "{}.mp4".format(args["type"])])
outp = os.path.sep.join([args["output"], args["type"]])
if not os.path.exists(outp):
    os.makedirs(outp)

# load video from disk
print("[INFO] loading video {}...".format(inp))
cap = cv2.VideoCapture(inp)
read = 0
saved = 0

# loop over frames from the video file stream
while True:
	# grab the frame from the file
	ret, frame = cap.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not ret:
		break

	# increment the total number of frames read thus far
	read += 1

	# check to see if we should process this frame
	if read % args["skip"] != 0:
		continue

	# grab the frame dimensions and construct a blob from the frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(image=cv2.resize(frame, (300, 300)), scalefactor=1.0,
		size=(300, 300), mean=(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()
	
	# print(np.shape(blob))
	# print(detections[0,0,1,:])
	# print(np.shape(detections))
	# break

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]

			# write the frame to disk
			p = os.path.sep.join([outp, "{}.png".format(saved)])
			cv2.imwrite(p, face)
			saved += 1
			print("[INFO] saved {} to disk".format(p))

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()