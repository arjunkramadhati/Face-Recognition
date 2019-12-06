
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
counter=0
while(True):
    _,frame = vs.read()
    
    if not facedetected or (counter%50==0):
        print("trying to detect")
        rects, landmarks = face_detect.detect_face(frame,net,predictor)
        if len(rects)>0:
            trackers=cv2.MultiTracker_create()
            print(len(rects))
            print("detected face")
            for i in rects:
                
                print(i)
                rect=(int(i[0]),int(i[1]),int(i[2]-i[0]),int(i[3]-i[1]))
                tracker=cv2.TrackerCSRT_create()
                trackers.add(tracker,frame,rect)		
            facedetected=True

    if facedetected:
        #rects=rects[0]
        #tuple([int(r) for r in rects[0:4]])
        #tracker.init(frame,rects)
        (success,box)=trackers.update(frame)
        if success:
            for b in box:
                (x,y,w,h)=[int(v) for v in b]
                cv2.rectangle(frame,(x,y),(w+x,h+y),(0,255,0))
            print("tracking")
        else:
            print("setting false")
            facedetected=False
    
    cv2.imshow("frame",frame)
    counter=counter+1
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        bb=cv2.selectROI("frame",frame,fromCenter=False)



