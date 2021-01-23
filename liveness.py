# USAGE
# python liveness.py

# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# from av import VideoFrame

class Liveness(object):
	def __init__(self, frame):
		# load our serialized face detector from disk
		print("[INFO] loading face detector...")
		protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
		modelPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])

		self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

		# load the liveness detector model and label encoder from disk
		print("[INFO] loading liveness detector...")
		self.model = load_model("liveness.model")
		self.le = pickle.loads(open("le.pickle", "rb").read())

		# initialize the video stream and allow the camera sensor to warmup
		print("[INFO] starting video stream...")
		# self.video = VideoStream(src=0).start()
		self.frame = frame
		self.label = "no input detected"
		# time.sleep(2.0)

	def __del__(self):
		# do a bit of cleanup
		cv2.destroyAllWindows()
		# self.video.stop()
	
	def getLabel(self):
		return self.label

	def setLabel(self, inputLabel):
		self.label = inputLabel

	def start(self):
		# loop over the frames from the video stream
		# while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 600 pixels

		print("[INFO] starting categorization...")

		# frame = vs.read()
		frame = self.frame
		frame = imutils.resize(frame, width=600)

		print("[INFO] starting dnn blobbing...")

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))

		print("[INFO] starting input blob in neural network...")
		# pass the blob through the network and obtain the detections and
		# predictions
		self.net.setInput(blob)
		detections = self.net.forward()

		print("[INFO] starting loop over detections...")
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > 0.5:

				# compute the (x, y)-coordinates of the bounding box for
				# the face and extract the face ROI
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the detected bounding box does fall outside the
				# dimensions of the frame
				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(w, endX)
				endY = min(h, endY)

				# extract the face ROI and then preproces it in the exact
				# same manner as our training data
				print("[INFO] extracting face ROI")
				face = frame[startY:endY, startX:endX]
				face = cv2.resize(face, (32, 32)) # this crashes for some reason
				face = face.astype("float") / 255.0 # this creates a problem
				face = img_to_array(face)
				face = np.expand_dims(face, axis=0)

				# pass the face ROI through the trained liveness detector
				# model to determine if the face is "real" or "fake"
				print("[INFO] passing face ROI")
				preds = self.model.predict(face)[0]
				j = np.argmax(preds)
				label = self.le.classes_[j]

				# # self.face = face
				# self.label = label
				self.setLabel(label)
				print("[INFO] current value of label is %s" % self.label)
				
				# draw the label and bounding box on the frame
				print("[INFO] drawing label and bounding box")
				label = "{}: {:.4f}".format(label, preds[j])
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)

		# show the output frame and wait for a key press
		# cv2.imshow("Frame", frame)
		# key = cv2.waitKey(1) & 0xFF

		print("[INFO] returning frame")
		return frame
		# return self.label

		# cv2.destroyAllWindows()
		# self.video.stop()