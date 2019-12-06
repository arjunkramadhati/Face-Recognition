from datetime import date
import time
def logfileEntry(stringent):
    Day = date.today()
    Day = str(Day)
    Time = time.strftime("%I:%M:%S %p",time.localtime())
    f1=open("log/"+Day+".txt","a+")
    f1.write("\n"+Day+" "+Time+" / "+str(stringent))
    f1.close()
