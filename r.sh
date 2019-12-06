#!/usr/bin/python
from subprocess import Popen
import sys

while True:
    print("starting")
    p=Popen(["python3", "/home/sigsenz/Desktop/FaceRec-master_new_FR/BGL_Face_Algo.py"])
    p.wait()
