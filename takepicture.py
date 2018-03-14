#!/usr/bin/python3

from gpiozero import LED
import datetime
import time
import os
import subprocess

directory = "/home/pi/"+str(datetime.datetime.now().strftime('%Y-%m-%d')+"/")
if not os.path.isdir(directory):
    print("created folder at "+ directory)
    os.makedirs(directory)
else:
    print("folder already existent at "+ directory)
led = LED(3)
while(True):
    led.on()
    #take picture
    subprocess.call("raspistill -n -w 1280 -h 720 -o "+directory + datetime.datetime.now().strftime('%H_%-M')+".jpg -v -a 12", shell=True)
    led.off()
    #time.sleep(10)#debug
    time.sleep(30*60)#30 min
    