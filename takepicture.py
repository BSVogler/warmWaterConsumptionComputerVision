#!/usr/bin/python3

from gpiozero import LED
import datetime
import time
import os
import subprocess


led = LED(3)
while(True):
    currentTime = datetime.datetime.now()
    directory = "/home/pi/"+str(currentTime.strftime('%Y-%m-%d')+"/")
    if not os.path.isdir(directory):
        print("created folder at "+ directory)
        os.makedirs(directory)
    led.on()
    #take picture
    filename = directory + currentTime.strftime('%H_%-M')+".jpg"
    subprocess.call("raspistill -n -w 1280 -h 720 -o "+filename+" -v -a 12", shell=True)
    print("took picture "+filename)
    led.off()
    #time.sleep(10)#debug
    time.sleep(30*60)#30 min
    