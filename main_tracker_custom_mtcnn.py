import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect_custom import MTCNNDetect
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

from gstreamer import *
from streaming.server import StreamingServer
#from gstreamer import Display, run_gen

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
path_prototxt="/home/sigsenz/Desktop/FaceRec-master_new_FR/deploy.prototxt.txt"
path_model="/home/sigsenz/Desktop/FaceRec-master_new_FR/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(path_prototxt,path_model)

#reading pretrained model for landmark features
path_shape_predictor="/home/sigsenz/Desktop/FaceRec-master_new_FR/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(path_shape_predictor)


def get_color(id):
    return svg.rgb((255,0,0))

def size_em(length):
    return '%sem' % str(0.6 * length)


class Video():
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self,render_size=Size(640,480), inference_size=Size(300,300), loop=False, port=5600):
        """Summary
        webcam:appsrc ! videoconvert ! v4l2h264enc extra-controls=\"encode,frame_level_rate_control_enable=1,video_bitrate=8380416\" ! h264parse !
        Args:
            port (int, optional): UDP port
        """
        
        Gst.init(None)
        #server initialization

        self._layout = make_layout(inference_size, render_size)
        self._loop = loop
        self._thread = None
        self.render_overlay = None
        self._obj=None
        self.counter=0
        #rtsp gstreamer
        self.port = port
        self._frame = None
        self.isRecognising = False
        self.trackers=[]
        self.labels=[]
        

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
        self.video_sink = None
        
        self.h264_sink = None
        #self.run()

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
        '''command="""rtspsrc location=rtsp://admin:senz25118@192.168.1.101 latency=0 ! queue ! application/x-rtp,encoding--name=h264,media=video,framerate=10,width=640,height=480 ! rtph264depay ! avdec_h264 ! tee name=t
t. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! x264enc threads=4 aud=False tune=zerolatency speed-preset=ultrafast bitrate=1000 key-int-max=5 ! video/x-h264,profile=baseline ! h264parse ! video/x-h264,alignment=nal,stream-format=byte-stream ! appsink sync=False drop=False max-buffers=1 emit-signals=True name=h264sink
t. ! queue max-size-buffers=1 ! decodebin ! glfilterbin filter=glcolorscale ! video/x-raw,format=RGBA,height=225,width=300 ! videoconvert ! video/x-raw,format=RGB,height=225,width=300 ! videobox autocrop=True ! video/x-raw,height=300,width=300 ! appsink sync=False drop=True max-buffers=1 emit-signals=True name=appsink"""'''
        command="""v4l2src device=/dev/video0 ! video/x-raw, height=480, width=640, framerate=10/1 ! tee name=t
t. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! x264enc threads=4 aud=False tune=zerolatency speed-preset=ultrafast bitrate=1000 key-int-max=5 ! video/x-h264,profile=baseline ! h264parse ! video/x-h264,alignment=nal,stream-format=byte-stream ! appsink sync=False drop=False max-buffers=1 emit-signals=True name=h264sink
t. ! queue max-size-buffers=1 ! decodebin ! glfilterbin filter=glcolorscale ! video/x-raw,format=RGBA,height=225,width=300 ! videoconvert ! video/x-raw,format=RGB,height=225,width=300 ! videobox autocrop=True ! video/x-raw,height=300,width=300 ! appsink sync=False drop=True max-buffers=1 emit-signals=True name=appsink"""
        print(command)
        self.video_pipe = Gst.parse_launch(command)
        print("launched")
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink')
        self.h264_sink = self.video_pipe.get_by_name('h264sink')

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

    def frame(self):
        """ Get Frame

        Returns:
            iterable: bool and image frame, cap.read() output 
        """
        return self._frame

    def remove_frame(self):
        self._frame=None

    def frame_available(self):
        """Check if frame is available

        Returns:
            bool: true if frame is available
        """
        return type(self._frame) != type(None)

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

        def on_buffer(data, _):
            if self._obj is not None:
                self._obj.write(data)

        def render_overlay(tensor, layout, command):
            if self.render_overlay:
                self.render_overlay(tensor, layout, command)
            return None

        self.video_sink.connect('new-sample', self.callback)
        self.h264_sink.connect('new-sample', new_sample_callback(on_buffer))

    def callback(self, sink):
        self.counter=self.counter+1
        sample = sink.emit('pull-sample')
        frame = self.gst_to_opencv(sample)
        
        if len(self.trackers)>0:
            rects_tracker_overlay=[]
            rect_tracker_overlay=[]
            for (label,tracker) in zip(self.labels, self.trackers):
                update=tracker.update(frame)
                if update>4:
                    pos=tracker.get_position()
                    rect_tracker_overlay=[int(pos.left()),int(pos.top()),int(pos.right()),int(pos.bottom())]
                    rects_tracker_overlay.append(rect_tracker_overlay)
            objs = [convert(obj,self.labels[i][:]) for (i,obj) in enumerate(rects_tracker_overlay)]
            print("Tracking")    
            self.render_overlay(overlay("cpu",objs,get_color,0.1,0.2,self._layout))
        
        
        if self.counter%50==0 or len(self.trackers)==0:
            print("detecting")
            self.trackers=[]
            self.labels=[]
            self.counter=0
            #self.recog_face(frame)
            if not self.isRecognising:
                self.isRecognising = True
                thread1=threading.Thread(target=self.recog_face,args=(frame,net,predictor,))
                thread1.start()
            #print("recognized")
            #print(frame.shape)
        

        return Gst.FlowReturn.OK


    def recog_face(self,frame,net,predictor):
        rects, landmarks = face_detect.detect_face(frame,net,predictor);#min face size is set to 80x80
        #rects, landmarks = face_detect.detect_face(frame,15);
        #print(rects)
        #print(landmarks)
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            #print(i)
            aligned_face, face_pos = aligner.align(160,frame,landmarks[:,i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
                #print("Face Detected")
            else: 
                print("Align face failed") #log        
        if(len(aligns) > 0):
            features_arr = extract_feature.get_features(aligns)
            recog_data = findPeople(features_arr,positions)
            
            
            for (i,rect) in enumerate(rects):
                rectangle_tracker=dlib.rectangle(int(rect[0]),int(rect[1]),int(rect[2]),int(rect[3]))
                tracker=dlib.correlation_tracker()
                tracker.start_track(frame,rectangle_tracker)
                self.trackers.append(tracker)
                self.labels.append(recog_data[i][:])
                if recog_data[i][0] !="Unknown":
                    now=datetime.datetime.now()
                    print("Detected: ", recog_data[i][0])
                    print(now.strftime("%Y-%m-%d %H:%M:%S"))
                #cv2.rectangle(frame,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),(255,0,0)) #draw bounding box for the face
                #cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(int(rect[0]),int(rect[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            
            objs = [convert(obj,recog_data[i][:]) for (i,obj) in enumerate(rects)]    
                
            self.render_overlay(overlay("cpu",objs,get_color,0.1,0.2,self._layout))
        else:
            self.render_overlay(None)
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

def camera_recog():
    print("[INFO] camera sensor warming up...")
    video = Video(); #get input from webcam
    #svout=cv2.VideoWriter("/home/vijayalaxmi/Documents/Face Recognition/FaceRec-master/output.avi",0,10,(640,480))
    #vs = cv2.VideoCapture(1)
    #print("b4 server")
    with StreamingServer(video, 10000) as server:
        def render_overlay(overlay):
            #print(overlay)
            #overlay = gen.send((tensor, layout, command))
            server.send_overlay(overlay if overlay else EMPTY_SVG)
            #pass

        video.render_overlay = render_overlay
        signal.pause()

def convert(obj, labels):
    x0=obj[0]
    y0=obj[1]
    x1=obj[2]
    y1=obj[3]
    return Object(id=labels[0],
                  label=labels[0],
                  score=labels[1],
                  bbox=BBox(x=x0, y=y1, w=x1-x0, h=y1-y0))

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
    f = open('./facerec_128D.txt','r')
    data_set = json.loads(f.read());
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

def overlay(title, objs, get_color, inference_time, inference_rate, layout):
    #print(layout.window)
    x0, y0, width, height = layout.window
    font_size = 0.03 * height

    defs = svg.Defs()
    defs += CSS_STYLES

    doc = svg.Svg(width=width, height=height,
                  viewBox='%s %s %s %s' % layout.window,
                  font_size=font_size, font_family='monospace', font_weight=500)
    doc += defs

    for obj in objs:
        percent = int(obj.score)
        if obj.label:
            caption = '%d%% %s' % (percent, obj.label)
        else:
            caption = '%d%%' % percent

        x, y, w, h = obj.bbox.scale(layout.render_size[0]/layout.inference_size[0],layout.render_size[1]/layout.inference_size[1])
        color = get_color(obj.id)

        doc += svg.Rect(x=x, y=y, width=w, height=h,
                        style='stroke:%s' % color, _class='bbox')
        doc += svg.Rect(x=x, y=y+h ,
                        width=size_em(len(caption)), height='1.2em', fill=color)
        t = svg.Text(x=x, y=y+h, fill='black')
        t += svg.TSpan(caption, dy='1em')
        doc += t

    ox = x0 + 20
    oy1, oy2 = y0 + 20 + font_size, y0 + height - 20

    # Title
    if title:
        doc += svg.Rect(x=0, y=0, width=size_em(len(title)), height='1em',
                        transform='translate(%s, %s) scale(1,-1)' % (ox, oy1), _class='back')
        doc += svg.Text(title, x=ox, y=oy1, fill='white')

    # Info
    lines = [
        'Objects: %d' % len(objs),
        'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time)
    ]

    for i, line in enumerate(reversed(lines)):
        y = oy2 - i * 1.7 * font_size
        doc += svg.Rect(x=0, y=0, width=size_em(len(line)), height='1em',
                       transform='translate(%s, %s) scale(1,-1)' % (ox, y), _class='back')
        doc += svg.Text(line, x=ox, y=y, fill='white')

    return str(doc)



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
    face_detect = MTCNNDetect(); #scale_factor, rescales image for faster detection
    main(args);
