activate_this = "/home/sigsenz/.virtualenvs/deep_learning/bin/activate_this.py"
execfile(activate_this, dict(__file__=activate_this))

from subprocess import Popen
import sys

while True:
    print("starting")
    p=Popen(["python", "BGL_Face_Algo.py"])
    p.wait()
