import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import time
import numpy as np
import gi
import dlib
import logging
import svg
import threading
import datetime
import loging
# from face_rec import FaceRecog
import subprocess
from gstreamer import *
#from streaming.server import StreamingServer
#from gstreamer import Display, run_gen
import os
gi.require_version('Gst', '1.0')
from gi.repository import Gst

EMPTY_SVG = str(svg.Svg())

CSS_STYLES = str(svg.CssStyle({'.back': svg.Style(fill='black',
                                                  stroke='black',
                                                  stroke_width='0.5em'),
                               '.bbox': svg.Style(fill_opacity=0.0,
                                                  stroke_width='0.1em')}))

Object = collections.namedtuple('Object', ('id', 'label', 'score', 'bbox'))
Object.__str__ = lambda self: 'Object(id=%d, label=%s, score=%.2f, %s)' % self

BBox = collections.namedtuple('BBox', ('x', 'y', 'w', 'h'))
BBox.area = lambda self: self.w * self.h
BBox.scale = lambda self, sx, sy: BBox(x=self.x * sx, y=self.y * sy,
                                       w=self.w * sx, h=self.h * sy)
BBox.__str__ = lambda self: 'BBox(x=%.2f y=%.2f w=%.2f h=%.2f)' % self

TIMEOUT = 10 #10 seconds

###reading model for face detection
'''path_prototxt="/home/sigsenz/Documents/FaceRec-master_new_FR/deploy.prototxt.txt"
path_model="/home/sigsenz/Documents/FaceRec-master_new_FR/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(path_prototxt,path_model)

reading pretrained model for landmark features '''

path_shape_predictor="/home/sigsenz/Documents/FaceRec-master_new_FR/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(path_shape_predictor)


def get_color(id):
    return svg.rgb((0,255,0))

def size_em(length):
    return '%sem' % str(0.6 * length)

uids=['1','2','3','4','5','6','7']
class Video():

    def __init__(self,render_size=Size(640,480), inference_size=Size(640,480), loop=False, port=5600):
        """Summary
        webcam:appsrc ! videoconvert ! v4l2h264enc extra-controls=\"encode,frame_level_rate_control_enable=1,video_bitrate=8380416\" ! h264parse !
        Args:
            port (int, optional): UDP port
        """
        
        Gst.init(None)
        #server initialization
        super(Video,self).__init__()

        self._layout = make_layout(inference_size, render_size)
        self._loop = loop
        self._thread = None
        self.render_overlay = None
        self.render_face=None
        self._obj=None
        self.counter=0
        #rtsp gstreamer
        self.port = port
        self._frame = None
        self.isRecognising = False
        self.trackers=[]
        self.labels=[]
        self.labeltracker=[]
        self.labeltrackercopy=[]
        self.isRecognisingSecond=False
        self.reclist=[]
        self.uklabels=[]
        self.sleeping=False
        self.countertwo=0
        self.firstface=False
        self.c=0
        self.start_time=time.time()
        self.countrue=0
        self.serversent=True
        #self.f=cv2.imread("1.jpg")
        print("Video object created")        
        threadRst=threading.Thread(target=self.restart_service, args=(self.start_time,))
        threadRst.start()

        #udpsrc port=5600 ! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert ! appsink emit-signals=true sync=false max-buffers=2 drop=true
        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'rtspsrc location=rtsp://admin:senz25118@192.168.123.102 latency=0 ! queue '
        #self.video_source = "v4l2src device=/dev/video0 do-timestamp=true ! video/x-raw, width=640, height=480, framerate=15/1 ! videoconvert ! appsink"
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp,encoding--name=h264,media=video,framerate=10,width=640,height=480 ! rtph264depay ! avdec_h264 '
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! queue max-size-buffers=1 ! decodebin ! glfilterbin filter=glcolorscale ! video/x-raw,format=RGBA,height=225,width=300 ! videoconvert ! video/x-raw,format=RGB,height=225,width=300 ! videobox autocrop=True ! video/x-raw,height=300,width=300 '
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink sync=False drop=True max-buffers=1 emit-signals=True name=appsink'
        self.h264_sink_conf='! queue max-size-buffers=1 leaky=downstream ! videoconvert ! x264enc threads=4 aud=False tune=zerolatency speed-preset=ultrafast bitrate=1000 key-int-max=5 ! video/x-h264,profile=baseline ! h264parse ! video/x-h264,alignment=nal,stream-format=byte-stream ! appsink sync=False drop=False max-buffers=1 emit-signals=True name=h264sink'
        self.video_pipe = None
        self.video_sink1 = None
        
        self.h264_sink = None
        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]


        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)

        command=""
        command="""v4l2src device=/dev/video0 ! video/x-raw, height=480, width=640, framerate=10/1 ! tee name=t
t. ! queue max-size-buffers=1 ! decodebin ! glfilterbin filter=glcolorscale ! video/x-raw,format=RGBA,height=480,width=640 ! videoconvert ! video/x-raw,format=RGB,height=480,width=640 ! videobox autocrop=True ! video/x-raw,height=480,width=640 ! appsink sync=False drop=True max-buffers=1 emit-signals=True name=appsink1"""

        #print(command)
        self.video_pipe = Gst.parse_launch(command)
        print("launched")
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink1 = self.video_pipe.get_by_name('appsink1')
        #self.h264_sink = self.video_pipe.get_by_name('h264sink')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

   

    def run(self):
        """ Get frame to update _frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf,
                self.h264_sink_conf
            ])

        
        #if self.video_sink1 is not None:
            #self.video_sink1.connect('new-sample', self.callback1)
            #print("Hiii")


    def callback1(self):
        sink=self.video_sink1
        #print("HIIIIIIIII")
        
        self.counter += 1
        sample = sink.emit('pull-sample')
        frame = self.gst_to_opencv(sample)
        framer=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        if self.firstface==False:
            self.c+=1
            print("false pic")
            path="/home/sigsenz/Documents/FaceRec-master_new_FR/" + str(self.c) + ".jpg"
            framer=cv2.imread(path)
            if self.c==10:
                self.c=0

        if self.counter%10==0 :
            #print("detecting")
            print(self.isRecognising)
            self.trackers=[]
            self.labels=[]
            self.counter=0
            self.labeltracker=[]
            #self.recog_face(frame)
            if self.isRecognising == False:
                if self.serversent==True:
                    self.serversent=False
                    threadserver=threading.Thread(target=self.serversend)
                    threadserver.start()
                self.isRecognising = True
                print("Analysing frame")
                thread1=threading.Thread(target=self.recog_face,args=(framer,))
                thread1.start()
            #print("recognized")
            #print(frame.shape)


        return Gst.FlowReturn.OK


    def restart_service(self,starttime):
        while True:
            now=time.time()
            if now - starttime >=120 and self.firstface==False:
                os.execl(sys.executable, sys.executable, *sys.argv)
                break
            if self.firstface==True:
                print("Killing firstface watch thread")
                break


    def serversend(self):
        os.system("php /home/sigsenz/Documents/FaceRec-master_new_FR/client1.php")
        self.serversent=True

 
    def recog_face(self,framer):

        #rects, landmarks = face_detect.detect_face(frame,net,predictor);#min face size is set to 80x80
        rects, landmarks = face_detect.detect_face(framer,20);
        #print(rects)
        #print(landmarks)
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            #print(i)
            aligned_face, face_pos = aligner.align(160,framer,landmarks[:,i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
                #print("Face Detected")
            else: 
                print("Align face failed") #log        
        if(len(aligns) > 0):
            features_arr = extract_feature.get_features(aligns)
            recog_data = findPeople(features_arr,positions)
            loging.logfileEntry(str(recog_data))
            f=open("/home/sigsenz/Documents/FaceRec-master_new_FR/status.txt","w+")
            for (i,rect) in enumerate(rects):
                
                if recog_data[i][0] !="Unknown":
                    if self.firstface==False:
                        print("Setting true")
                        self.firstface=True
                    print("Detected: ", recog_data[i][0])
                    print("matching; ", recog_data[i][1])

                    
                    #print(now.strftime("%Y-%m-%d %H:%M:%S"))

                elif recog_data[i][0] =="Unknown":
                    if self.firstface==False:
                        print("Setting true")
                        self.firstface=True
                    print("Unknown person: Not tracking")
                    print("matching : ",recog_data[i][1])
                f.write(str(recog_data[i][0]) + "\n")
            f.close()
   
                #os.system("php /home/sigsenz/Desktop/FaceRec-master_new_FR/client1.php")
        if(len(aligns) <= 0):
            f=open("/home/sigsenz/Documents/FaceRec-master_new_FR/status.txt","w+")
            f.write("No person")
            f.close()
            #os.system("php /home/sigsenz/Desktop/FaceRec-master_new_FR/client1.php")
        self.isRecognising = False
        #self.trackers=[]
        #self.labels=[]

    @property
    def resolution(self):
        #print("======================================")
        #print(self._layout.render_size)
        return self._layout.render_size

    def request_key_frame(self):
        pass

  
    def start_recording(self, obj, format, profile, inline_headers, bitrate, intra_period):
        self._obj=obj
        self.run()
        '''def on_buffer(data, _):
            obj.write(data)

        def render_overlay(tensor, layout, command):
            if self.render_overlay:
                self.render_overlay(tensor, layout, command)
            return None'''

        '''signals = {
          'h264sink': {'new-sample': gstreamer.new_sample_callback(on_buffer)},
        }

        pipeline = self.make_pipeline(format, profile, inline_headers, bitrate, intra_period)

        self._thread = threading.Thread(target=gstreamer.run_pipeline,
                                        args=(pipeline, self._layout, self._loop,
                                              render_overlay, gstreamer.Display.NONE,
                                              False, signals))
        self._thread.start()'''
        #pass
    
    def stop_recording(self):
        '''gstreamer.quit()
        self._thread.join()'''
        self._obj=None
        #pass

    def make_pipeline(self, fmt, profile, inline_headers, bitrate, intra_period):
        raise NotImplemented


def main(args):
    mode = args.mode
    if(mode == "camera"):
        camera_recog()
    elif mode == "input":
        create_manual_data();
    else:
        raise ValueError("Unimplemented mode")

data_set = {}

def camera_recog():
    global data_set
    f = open('/home/sigsenz/Documents/FaceRec-master_new_FR/facerec_128D.txt','r')
    data_set = json.loads(f.read());
    print("[INFO] camera sensor warming up...")
    video = Video()
    #print("Hi1221")
    #video.run()
    #signal.pause()
    while True:
        video.callback1()

def create_manual_data():
    video = Video(); #get input from webcam
    print("Please input new user ID:")
    new_name = input(); #ez python input()
    f = open('./facerec_128D.txt','r');
    data_set = json.loads(f.read());
    person_imgs = {"Left" : [], "Right": [], "Center": []};
    person_features = {"Left" : [], "Right": [], "Center": []};
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");
    while True:
        if not video.frame_available():
            #print("not available")
            continue
        frame = video.frame()
        rects, landmarks = face_detect.detect_face(frame,net,predictor);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160,frame,landmarks[:,i]);
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                cv2.imshow("Captured face", aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    for pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = person_features;
    f = open('./facerec_128D.txt', 'w');
    f.write(json.dumps(data_set))



def findPeople(features_arr, positions, thres = 0.6, percent_thres = 70):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    global data_set
    global uids
    returnRes = [];

    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance;
                    result = person;
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
    return returnRes 




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    args = parser.parse_args(sys.argv[1:]);
    
    #creating objects
    FRGraph = FaceRecGraph();
    MTCNNGraph = FaceRecGraph();
    aligner = AlignCustom();
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(MTCNNGraph,scale_factor=2); #scale_factor, rescales image for faster detection

    main(args);

