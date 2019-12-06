# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import dlib
import numpy as np
import argparse
import imutils
import time
import cv2

shape=[]
box=[]
shape_new=[]
landmarks=np.array([])

class MTCNNDetect_custom(object):

    def landmarks(self,frame,predictor,reclist):
        lands=landmark(frame,predictor,reclist)
        return lands

    def detect_face(self, frame, net,predictions):
        global box
        global landmarks
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
	# predictions
        net.setInput(blob)
        detections = net.forward()
        rect=[]
        for i in range(0, detections.shape[2]):
	    # extract the confidence (i.e., probability) associated with the
	    # prediction
            confidence = detections[0, 0, i, 2]

	    # filter out weak detections by ensuring the `confidence` is
	    # greater than the minimum confidence
            if confidence > 0.5:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
                box = list(detections[0, 0, i, 3:7] * np.array([w, h, w, h]))
                box.append(confidence*100)
                rect.append(box)
        landmarks=landmark(frame,predictions,rect)
        #print(box)
        return rect,landmarks

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
    return coords

def landmark(frame, predictor,rect):
    global shape
    global shape_new
    shape_con=[]
    image = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (i, rect) in enumerate(rect):
	    # determine the facial landmarks for the face region, then
	    # convert the facial landmark (x, y)-coordinates to a NumPy
	    # array
        #print(type(rect))
        #print(rect)
        rect=dlib.rectangle(left=int(rect[0]),top=int(rect[1]),right=int(rect[2]),bottom=int(rect[3]))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #shape_new=[(shape[36][0]+shape[39][0])/2,(shape[36][1]+shape[39][1])/2,(shape[42][0]+shape[45][0])/2,(shape[42][1]+shape[45][1])/2,shape[30][0],shape[30][1],shape[48][0],shape[48][1],shape[54][0],shape[54][1]]
        shape_new=[(shape[36][0]+shape[39][0])/2,(shape[42][0]+shape[45][0])/2,shape[30][0],shape[48][0],shape[54][0],(shape[36][1]+shape[39][1])/2,(shape[42][1]+shape[45][1])/2,shape[30][1],shape[48][1],shape[54][1]]
        shape_con.append(shape_new)
    landmark_full=np.array(shape_con)
    landmark_full=np.transpose(landmark_full)	
    return landmark_full

   
   
        
        



                
