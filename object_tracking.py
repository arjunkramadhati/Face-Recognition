
import cv2
import imutils
from mtcnn_detect import MTCNNDetect
import numpy as np
import dlib


###reading model for face detection
path_prototxt="/home/sigsenz/Desktop/FaceRec-master_new_FR/deploy.prototxt.txt"
path_model="/home/sigsenz/Desktop/FaceRec-master_new_FR/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(path_prototxt,path_model)

#reading pretrained model for landmark features
path_shape_predictor="/home/sigsenz/Desktop/FaceRec-master_new_FR/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(path_shape_predictor)

face_detect = MTCNNDetect();

vs = cv2.VideoCapture(0); #get input from webcam

#trackers=cv2.MultiTracker_create()
facedetected=False
bb=None
while(True):
    _,frame = vs.read()
    
    if not facedetected:
        print("trying to detect")
        rects, landmarks = face_detect.detect_face(frame,net,predictor)
        if len(rects)>0:
            print(len(rects))
            print("detected face")
            #rects=rects[0]
            #print(rects[0])
            #rects=(int(rects[0]),int(rects[1]),int(rects[2]),int(rects[3]))
            rect=dlib.rectangle(int(rects[0]),int(rects[1]),int(rects[2]),int(rects[3]))
            tracker=dlib.correlation_tracker()
            tracker.start_track(frame,rect)		
            facedetected=True

    if facedetected:
        #rects=rects[0]
        #tuple([int(r) for r in rects[0:4]])
        #tracker.init(frame,rects)
        #(success,box)=tracker.update(frame)
        
        if traker.update(frame)>0.3:
            #traker.update(frame)
            pos=tracker.get_position()
            cv2.rectangle(frame,(int(pos.left()),int(pos.top())),(int(pos.right()),int(pos.bottom())),(0,255,0))
            print("tracking")
        else:
            print("setting false")
            facedetected=False
    
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        bb=cv2.selectROI("frame",frame,fromCenter=False)



