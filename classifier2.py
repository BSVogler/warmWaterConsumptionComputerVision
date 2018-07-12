import classifier as cf
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib
import math
import sys
import os
from scipy.misc import toimage
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from collections import namedtuple
import datetime
from scipy.ndimage.filters import gaussian_filter
path = "images/23_49.jpg"
#%%
yellowCorrection = (-20,-20)
blueCorrection = (20,-20)
rgbOrigPIL= Image.open(path)
rgbOrigPIL.load()
rgb = np.asarray(rgbOrigPIL, dtype="float32")[100:-300,200:-380,:]
hsv = matplotlib.colors.rgb_to_hsv(rgb)

#%% show saturation distribution
plt.hist(hsv[:,:,1].flatten(), bins=256)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
#%% Calcualte scale space
scaleSpaceImageHSV = matplotlib.colors.rgb_to_hsv(cf.scaleSpace(rgb,2))

#%% filter saturated colors and apply boundaries
saturated = np.zeros_like(rgb)
boundaries = [hsv.shape[0], 0, hsv.shape[1], 0]

for r in range(saturated.shape[0]):
    for c in range(saturated.shape[1]):
        if (scaleSpaceImageHSV[r,c,1] > 0.4 and scaleSpaceImageHSV[r,c,2] > 0.3):
            saturated[r,c] = rgb[r,c]
            if r < boundaries[0]:
                boundaries[0] = r
            if r > boundaries[1]:
                boundaries[1] = r
            if c < boundaries[2]:
                boundaries[2] = c
            if c > boundaries[3]:
                boundaries[3] = c

print(boundaries)
#only apply boundaries in x direction, this is needed because of the later rotation
saturated = saturated[:, boundaries[2]:boundaries[3], :]
cf.whiteBalance(sourceRGB=rgb, destRGB=saturated)
#scaleSpaceImageHSV = scaleSpaceImageHSV[:, boundaries[2]:boundaries[3], :]

display(Image.fromarray(np.uint8(saturated)))
hsv = hsv[:, boundaries[2]:boundaries[3], :]
rgb = rgb[:, boundaries[2]:boundaries[3], :]

#%% find rightmost yellow pixel
#yellow = matplotlib.colors.rgb_to_hsv((127,127,0))
saturatedHSV = matplotlib.colors.rgb_to_hsv(saturated)
#saturatedHSV = matplotlib.colors.rgb_to_hsv(cf.scaleSpace(matplotlib.colors.hsv_to_rgb(saturatedHSV),2))
display(Image.fromarray(np.uint8(matplotlib.colors.hsv_to_rgb(saturatedHSV)))) 
yellowCenter = cf.findMaximumColor(saturatedHSV,(127,127,0),errormarginH=0.02,minS=0.3)
yellowCenter=(yellowCenter[0]+yellowCorrection[0],yellowCenter[1]+yellowCorrection[1])
blueCenter = cf.findMaximumColor(saturatedHSV,(0,102,147),errormarginH=0.2,minS=0.2)
blueCenter=(blueCenter[0]+blueCorrection[0],blueCenter[1]+blueCorrection[1])

#foundyellow = False
#for r in range(hsv.shape[0]):
#    for c in range(hsv.shape[1]):
#        if foundyellow and abs(hsv[r,c,0]-yellow[0]) < 0.1:
            
#        foundyellow=True
        
angle = -math.degrees(math.atan((abs(yellowCenter[0]-blueCenter[0]))/(abs(yellowCenter[1]-blueCenter[1]))))
print("Rotate by "+str(angle)+"°")
rotatedRGB = Image.fromarray(np.uint8(rgb))#0-1 numpy array to uint8 pillow
#highlight
draw = ImageDraw.Draw(rotatedRGB)
draw.rectangle((yellowCenter[1], yellowCenter[0], yellowCenter[1]+2, yellowCenter[0]+2), outline="red", fill=None)
draw.rectangle((blueCenter[1], blueCenter[0], blueCenter[1]+2, blueCenter[0]+2), outline="blue", fill=None)

rotatedRGB = rotatedRGB.rotate(angle, resample=Image.BILINEAR, center=yellowCenter)

rotatedRGB.rotate(angle, resample=Image.BILINEAR, center=yellowCenter)
display(rotatedRGB)

rotatedHSV = matplotlib.colors.rgb_to_hsv(rotatedRGB)

#%% find the interesting segment and fetch the digits
green = matplotlib.colors.rgb_to_hsv((0.56,0.64,0.39))
print(green)
green= green[0]
gaussHue = gaussian_filter(rotatedHSV[:,:,0], sigma=4) 
#display(Image.fromarray(np.uint8(gaussHue*255)))

r = yellowCenter[0]
left = yellowCenter[1]
distanceGreen = []
for offc in range(0, 80):
    c = yellowCenter[1]+40+offc
    if rotatedHSV[r][c][1]>0.2 and rotatedHSV[r][c][2]>0.5:
        distanceGreen.append(abs(gaussHue[r][c]-green))
    
left = yellowCenter[1]+40+np.argmin(distanceGreen)+10
plt.plot(distanceGreen)
plt.xlabel("Value")
plt.ylabel("Frequency")

r = blueCenter[0]-10
right = blueCenter[1]
distanceGreen = []
for offc in range(0, 150):
    c = blueCenter[1]-40-offc
    if rotatedHSV[r][c][1]>0.2 and rotatedHSV[r][c][2]>0.4:
        distanceGreen.append(abs(gaussHue[r][c]-green))
    
right = blueCenter[1]-40-np.argmin(distanceGreen)-10
plt.plot(distanceGreen)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
#interestingSegment = cf.findRectangle(rotatedRGB)
RectTupleClass = namedtuple("Rectangle", "left right top bottom")
interestingSegment = RectTupleClass(left, right, yellowCenter[0]-25, yellowCenter[0]+20)
print(interestingSegment)
draw = ImageDraw.Draw(rotatedRGB)
draw.rectangle(
        (interestingSegment.left,
        interestingSegment.top,
        interestingSegment.right,
        interestingSegment.bottom), outline="green", fill=None
    )
display(rotatedRGB)

#%%
digits = cf.segmentDigits(interestingSegment,np.asarray(rotatedRGB, dtype="float32"), draw)
for d in digits:
    display(Image.fromarray(np.uint8(d)))
cf.ocr(digits)

#%% look for local minimum in local difference histogram
#firstminIndex = np.amax(histogry)
#firstminIndex = 10
#for i in range(firstminIndex, histogry.shape[0]):
#    if histogry[i] > histogry[i-1]:
#        break
#threshold = histogrx[i]
#print(str(i)+" with value "+str(threshold))
#
#for r in range(diff.shape[0]):
#    for c in range(diff.shape[1]):
#        if (diff[r,c] > threshold):
#            rgb[r,c]=0
#display(Image.fromarray(np.uint8(rgb)))