# Face-Recognition Project

# Objective
The objective of the project was to build an accurate face recognition algorithm which could run in real time on resource contrained edge devices. This has to run in sync (on the same device) with a live video streaming server. 

# Method
Since the entire project is required to run on resource constrained edge devices, there is a real possibility of it crashing/hanging if proper care is not taken to optimise the scantly available resource. Additionally, the video stream has to be live streamed to a web port. Gstreamer was used to overcome the problems needed to achieve real time inferencing. With Gstreamer pipelines it is possible to build branched pipelines for multiple purposes. In this case, we need two pipelines:
1) Video stream to infer and recognise faces
2) Video stream for the live streaming by the http server. 
Coming to the method used to recognise faces a five step solution was adopted:
1) Detect faces from the frame
2) Draw bounding boxes around these faces and keep only the bounded area for next step
3) Align the face in each bounding box to make it a vertical alinged face
4) Extract the 128 face features
5) Compare the extracted features with the 128 face features of the known faces in the data base. The closest match is the recognised person

Face Detection: MTCNN Face Detection algorithm is used
Alignment: Custom built face aligner which can differentiate the left profile - center profile - right profile of the face
128 Face Feature Extractor: The state-of-the-art Tensorflow model of ResNet Inception V1 128 Face Feature Extractor 

NOTE: When the code is run the first tinme, it does not load the heavy face feature extractor until the first face is detected. After the first face is detected, the code loads the model into the RAM and this puts sudden load on the edge device and hence making things glitchy. To avoid this, the code first loops over images of very easily detectable faces. Hence, the code sort of pre loads the models even before the first face is detected from the actual live stream. 

# Training
Right from the start, emphasis was given on planning to make the face training process as easy as possible. The face registration tool is in a seperate github repository. 

# Implementation and Results
The face-rec algorithm is extremely accurate for frontal face recognition. Although the side face recognition is a bit glitchy, it proved to be better than most other algorithms. Training with more of the side face profiles will increase the accuracy. 

# EXPERIMENTAL: Face Tracking
To reduce the load on the already stressed processor, a face tracking module was used. The idea was to recognise the face and start tracking it, instead of recognising it always. The tracker was found to be very unreliable in lighting conditions that was not anything but ideal. 
