#!/usr/local/bin/python3

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib
import math
import sys
from scipy.misc import toimage
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from collections import namedtuple

def scaleSpace(image, sigma):
    #scale space
    #rgbScaleSpace=np.zeros_like(image)
    print("Scale space transformation")
    rgbScaleSpace = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0), order=0)
    
    return rgbScaleSpace

'''return rowTop, rowBottom, columnLeft, columnRight, angle, anchorpoint'''
def segmentation(imgRGB):
    print("Segmentation")
    imgRGB=imgRGB[:-20,:,:]#ignore border line
    scaledRGB = scaleSpace(imgRGB,10)
    #pilImage = Image.fromarray(np.uint8(scaledRGB), mode='RGB')
    #pilImage.show()
    #toimage(scaledRGB).show()
    scaledHSV = matplotlib.colors.rgb_to_hsv(scaledRGB)
    
    #calculate segmentation coordinates
    #blueAreaRGB = np.array([0.192,0.22,0.28])
    peak1 = findMaximumColor(scaledHSV, np.array([0.43*2*averageGrayValue,0.37*2*averageGrayValue,0*2*averageGrayValue]), errormarginH=0.05*2*averageGrayValue, minS=0.5*2*averageGrayValue)#yellow
    peak2 = findMaximumColor(scaledHSV, np.array([0.32*2*averageGrayValue,0.15*2*averageGrayValue,0.2*2*averageGrayValue]), errormarginH=0.155*2*averageGrayValue,minS=0.11*2*averageGrayValue)#red
    
    if peak1[0]-peak2[0]==0 and peak1[1]-peak2[1]==0:
        print("could not find markers")
        return
      
    angle = math.atan((peak2[0]-peak1[0])/(peak2[1]-peak1[1]))*180/math.pi 
    hypo= int(math.sqrt((peak2[0]-peak1[0])*(peak2[0]-peak1[0]) + (peak2[1]-peak1[1])*(peak2[1]-peak1[1])))+40
           
    return (peak1[0]-110,peak1[0]-35, peak1[1]-10, peak1[1]-10+hypo, angle+5, peak1)


def findMaximumColor(imgHSV, colorRGB, errormarginH, minS):
    print("calculating histogram to find the color peak pos" +str(colorRGB))
    colorHSV = matplotlib.colors.rgb_to_hsv(colorRGB);
    #search for valid pixels
    foundTarget = np.zeros((imgHSV.shape[0], imgHSV.shape[1]), dtype=bool)
    for x in range(imgHSV.shape[0]):
        for y in range(imgHSV.shape[1]):
            #if abs(scaledHSV[x][y][0]-yellowAreaHSV[0]) < 0.07 and scaledHSV[x][y][1] > 0.3 and scaledHSV[x][y][2]<0.9 and scaledHSV[x][y][2]>0.3:
            if abs(imgHSV[x][y][0]-colorHSV[0]) < errormarginH and imgHSV[x][y][1] > minS and imgHSV[x][y][2]>0.18*2*averageGrayValue:
               foundTarget[x][y] = True
               imgHSV[x][y][0] = colorHSV[0]
               imgHSV[x][y][1] = 1
               imgHSV[x][y][2] = 0.5
    #toimage(matplotlib.colors.hsv_to_rgb(imgHSV)).show()         

    #calculate histograms by projecting rows and coloumns
    bucketReduction = 30
    numBucketHeight = int(imgHSV.shape[0]/bucketReduction)+1
    sumtotal=0
    rdir = np.zeros(numBucketHeight)
    for r in range(imgHSV.shape[0]):
        lastSum = np.sum(foundTarget[r])
        sumtotal += lastSum
        rdir[int(r/bucketReduction)] += lastSum
    
    numBucketWidth= int(imgHSV.shape[1]/bucketReduction)+1
    cdir = np.zeros(numBucketWidth)
    for c in range(imgHSV.shape[1]):
        lastSum = np.sum(foundTarget[:,c])
        sumtotal += lastSum
        cdir[int(c/bucketReduction)] += lastSum
           
    if sumtotal==0:
        print("did not find a peak")
        return None
    '''     
    plt.plot(rdir)
    plt.plot(cdir)
    plt.xlabel('X-Coordinate')
    plt.grid(True)
    plt.show()
    '''

    rowMax = int(np.argmax(rdir) * bucketReduction)
    columnMax = int(np.argmax(cdir) * bucketReduction)
    print("Maimumx r,c:"+str((rowMax, columnMax)))
    return (rowMax, columnMax)
 
#performs some scans to find the line with the maximum derivative (edges)   
def findRectangle(imageRGB):
    imageRGB = scaleSpace(imageRGB,2.5)
    imageHSV = matplotlib.colors.rgb_to_hsv(imageRGB)
    
    #find line with highest sum of derivative
    # maxV = None#maximum values of global maximum
    # #for row in range(1,imageRGB.shape[0],10):
    #     lastV = imageHSV[row][0][2]
    #     sumderiv = 0
    #     histo = []
    #     for coloumn in range(1,imageRGB.shape[1]):
    #         fx = imageHSV[row][coloumn][2]
    #         deriv = fx - lastV
    #         sumderiv += abs(deriv)
    #         histo.append(deriv)
    #         lastV = fx
    #
    #     #found new maximum
    #     if maxV == None or sumderiv > maxV[0]:
    #         maxV = (sumderiv, row, histo) #sum of derivative of V, row, histogram of V
    #
    #
    # if maxV == None:
    #     print("no central line found")
    #     return
        
    RectTupleClass = namedtuple("Rectangle", "left right top bottom")
    #look for biggest change in this line of Saturation    
    horLineHSV = imageHSV[maxV[1],:]
    dS = np.diff(horLineHSV[:,1])
    
    plt.plot(dS)
    plt.xlabel('X-Coordinate')
    plt.show()
    
    leftBorder = 0
    rightBorder = len(horLineHSV)
    #scan horizontally
    for c in range(leftBorder,rightBorder-1):
        if leftBorder==0:#first occurence
            if dS[c] <= -0.027:
                leftBorder = c + 8 #better if looking for local minimum, currently only adding fixed distortion
        elif horLineHSV[c][0] < 0.84 and horLineHSV[c][0] > 0.15 and dS[c] >= 0.027 and rightBorder==len(horLineHSV):#not red, first occurence
            rightBorder= c + 5
            

    #scan vertically
    verLineHSV = imageHSV[:,int(len(horLineHSV)/2)]#middle
    dS = np.diff(verLineHSV[:,1])
    ddS = np.diff(dS)
    
    topBorder = 0
    bottomBorder = len(verLineHSV)
    localMinimum = False
    localMaximum = False
    for r in range(topBorder, bottomBorder-2):
        if topBorder == 0:#first occurence
            if dS[r] <= -0.027:
                localMinimum = True
            if localMinimum and ddS[r] > 0: #wait till it rises again
                topBorder = r
        else:
            if (bottomBorder == len(verLineHSV) #first occurence
                and verLineHSV[r][0] < 0.84 and verLineHSV[r][0] > 0.15 #not red, 
                and dS[r] >= 0.027
            ):
                localMaximum = True
            if localMaximum and ddS[r] < 0:
                bottomBorder= r
    
    #if no result was found, take the last
    if localMaximum and bottomBorder == len(verLineHSV):
        bottomBorder= r
    
    
    return RectTupleClass(leftBorder,rightBorder, topBorder, bottomBorder)

def rotate2Dvector(point, angle, origin=(0, 0)):
    cos_theta, sin_theta = math.cos(angle), math.sin(angle)
    x, y = point[0] - origin[0], point[1] - origin[1]#translate
    Point = namedtuple('Point', 'x y')
    return Point(x * cos_theta - y * sin_theta + origin[0],
            x * sin_theta + y * cos_theta + origin[1])

#performs ocr using tesseract, input is list of numpy images
#todo improve using https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality
def orc(digits):
    import pytesseract
    digitValue = 10000000 # value of the next digit
    asValue = 0 #resulting value
    for digit in digits:
        #digit = Image.fromarray(np.uint8(digit*255), mode='RGB')#0-1 numpy array to uint8 pillow
        digitasNumber = pytesseract.image_to_string(digit, config='-psm 10 digits')
        print(digitasNumber)
        if digitasNumber == "":
            digitasNumber = 0
        else:
            digitasNumber = int(digitasNumber)
        asValue += digitasNumber*digitValue
        digitValue = int(digitValue/10)#numerical stability
    asValue *= 0.001
    print(asValue)
    
if __name__ == '__main__':
    input = "./images/image.jpg"
    if len(sys.argv)>1:
        input = sys.argv[1]
    rgbOrigPIL= Image.open(input)
    rgbOrigPIL.load()
    rgbOrig = np.asarray( rgbOrigPIL, dtype="float32" )
    print("Input dimension: "+str(rgbOrig.shape))
    
    #use only a cut of the original pucture
    reductionWidth = 0.48
    reductionHeight = 0.1
    print(str(reductionWidth*100)+"%x"+str(reductionHeight*100)+"% image size reduction")
    rgb = np.array(rgbOrig[int(rgbOrig.shape[0]*reductionHeight/2):int(rgbOrig.shape[0]*(1-reductionHeight/2)), int(rgbOrig.shape[1]*reductionWidth/2):int(rgbOrig.shape[1]*(1-reductionWidth/2)),:], copy=True) 
    print("Scaled dimension: "+str(rgb.shape))
    #white balance so that the average value is as defined
    averageRGB = [rgb[:, :, i].mean() for i in range(3)]
    print("average RGB"+str(averageRGB))
    
    global averageGrayValue 
    averageGrayValue=0.5
    wbCorrection = np.divide([averageGrayValue, averageGrayValue, averageGrayValue], averageRGB)#average 0.5 rgb value
    print("WB: "+str(wbCorrection))
    rgb[:,:] = np.multiply(rgb[:,:], wbCorrection)
    
    #correctedRGB= matplotlib.colors.hsv_to_rgb(imgHSV)
    #pilImage = Image.fromarray(rgbOrig, mode='RGB')
    #rgb = np.clip(rgb,0,1)
    #rgb[:,:] = np.multiply(rgbOrig[:,:], [256,256,256])
    #pilImage = Image.fromarray(rgb, mode='RGB')#here only showing shit
    #pilImage.show()
    #toimage(rgb).show()#shows correctly

    imgHSV = matplotlib.colors.rgb_to_hsv(rgb)
    
    #normalize
    minV = np.min(imgHSV[:,:,2])
    imgHSV[:, :, 2] -= minV #minimum value at 0

    maxV = np.max(imgHSV[:,:,2])
    imgHSV[:, :, 2] /= maxV #maximum value is 1
    #averageV = imgHSV[:, :, 2].mean()
    #imgHSV[:, :, 2] *= averageGrayValue / averageV #scale that average is averageGrayValue
    correctedRGB= matplotlib.colors.hsv_to_rgb(imgHSV)
    
    #get segmentation cut
    segRange = segmentation(correctedRGB) 
    print("Segmentation: "+str(segRange))
    #toimage(correctedRGB).show()#show segment
    pilRGB = Image.fromarray(np.uint8(correctedRGB*255), mode='RGB')#0-1 numpy array to uint8 pillow
    rotationAnchor = (segRange[5][1],segRange[5][0])
    pilRGB = pilRGB.rotate(segRange[4], resample=Image.BILINEAR, center=rotationAnchor)
    #pilRGB.show()
    correctedRGB = np.asarray(pilRGB)
    
    #rgb = ndimage.interpolation.rotate(correctedRGB, segRange[4], reshape=False)#rotate using scipy ndimage            
    correctedRGB = correctedRGB[segRange[0]:segRange[1], segRange[2]:segRange[3], :]
    #toimage(correctedRGB).show()#show segment
    
    #rgb = rgb.reshape(rgb.shape[0]*rgb.shape[1], rgb.shape[2])
    #translate image by the difference of 
    draw = ImageDraw.Draw(rgbOrigPIL)
    #draw image size reduction
    draw.rectangle(
        (rgbOrig.shape[1]*reductionWidth/2,
        rgbOrig.shape[0]*reductionHeight/2,
        rgbOrig.shape[1]*(1-reductionWidth/2),
        rgbOrig.shape[0]*(1-reductionHeight/2)
        ), outline="blue", fill=None)
   
    #rotate point around anchor, bug: reduction is shorter because of rotation, current implementation does not shrink it
    rotationAnchor = (rgbOrig.shape[1]*reductionWidth/2 + rotationAnchor[0], rgbOrig.shape[0]*reductionHeight/2 +rotationAnchor[1])#from r,c to x,y
    rgbOrigPIL = rgbOrigPIL.rotate(segRange[4], resample=Image.BILINEAR, center=rotationAnchor)
    cornerAbsolute = (rgbOrig.shape[1]*reductionWidth/2 + segRange[2], rgbOrig.shape[0]*reductionHeight/2+ segRange[0])
    cornerSegment = rotate2Dvector(cornerAbsolute, segRange[4]*math.pi/180.0, rotationAnchor)
    
    #draw things
    segmentWidth = segRange[3] - segRange[2]
    segmentHeight= segRange[1] - segRange[0]
    draw = ImageDraw.Draw(rgbOrigPIL)
    
    #draw segmentation
    draw.rectangle(
        (cornerSegment.x,
        cornerSegment.y ,
        cornerSegment.x + segmentWidth,
        cornerSegment.y + segmentHeight
        ), outline="red", fill=None)
    
    whiteRect = findRectangle(correctedRGB)
    print(whiteRect)
    #draw central line
    #draw.line(
    #    (cornerSegment.x,
    #    cornerSegment.y + maxLine[0],
    #    cornerSegment.x + segmentWidth,
    #    cornerSegment.y + maxLine[0])
    #)
    
    #draw borders of white segmentation block
    draw.line(
        (cornerSegment.x + whiteRect.left,
        cornerSegment.y + whiteRect.top,
        cornerSegment.x + whiteRect.left + whiteRect.right,
        cornerSegment.y + whiteRect.top)
    )
    draw.line(
       (cornerSegment.x + whiteRect.left,
        cornerSegment.y + whiteRect.bottom,
        cornerSegment.x + whiteRect.left + whiteRect.right,
        cornerSegment.y + whiteRect.bottom)
    )
    
    #segment digits from rectangle
    digits = []
    numberOfDigits = 8
    width = whiteRect.right - whiteRect.left
    if width < 5:
        print("Width of segment is too small. Segmentation failed.")
    stepSize = int(width / numberOfDigits)
    xLeftBorder = whiteRect.left
    for x in range(whiteRect.left, whiteRect.right, stepSize):
        draw.line(
            (cornerSegment.x + x,
            cornerSegment.y,
            cornerSegment.x + x,
            cornerSegment.y + segmentHeight)
        )
        if x > maxLine[1][0]:
            newDigit = correctedRGB[whiteRect.top : whiteRect.bottom, xLeftBorder+4 : x-2]#little bit of offset because of the border
            digits.append(newDigit)
            toimage(newDigit).show()
            xLeftBorder = x
        
    #show marked picture
    toimage(rgbOrigPIL).show()
    
    ocr(digits)        
    
    #toimage(rgb).show()