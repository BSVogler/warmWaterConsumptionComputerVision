#!/usr/local/bin/python3

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib
import math
import sys
from scipy.misc import toimage
from PIL import Image, ImageDraw
from collections import namedtuple
import datetime
from scipy.ndimage.filters import gaussian_filter

#returns nd-image
def scaleSpace(image, sigma):
    #scale space
    #rgbScaleSpace=np.zeros_like(image)
    print("Scale space transformation")
    rgbScaleSpace = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0), order=0)
    
    return rgbScaleSpace

def segmentation(imgRGB):
    '''
    return rowTop, rowBottom, columnLeft, columnRight, angle, anchorpoint
    :param imgRGB:
    :return:
    '''
    print("Segmentation")
    scaledRGB = scaleSpace(imgRGB,10)
    #pilImage = Image.fromarray(np.uint8(scaledRGB), mode='RGB')
    #pilImage.show()
    #toimage(scaledRGB).show()
    scaledHSV = matplotlib.colors.rgb_to_hsv(scaledRGB)
    
    # calculate segmentation coordinates
    #blueAreaRGB = np.array([0.192,0.22,0.28])
    peak1 = findMaximumColor(scaledHSV, np.array([0.43*2*averageGrayValue,0.37*2*averageGrayValue,0*2*averageGrayValue]), errormarginH=0.05*2*averageGrayValue, minS=0.5*2*averageGrayValue)#yellow
    peak2 = findMaximumColor(scaledHSV, np.array([0.32*2*averageGrayValue,0.15*2*averageGrayValue,0.2*2*averageGrayValue]), errormarginH=0.155*2*averageGrayValue,minS=0.11*2*averageGrayValue)#red
    
    if peak1[0]-peak2[0]==0 and peak1[1]-peak2[1]==0:
        print("could not find markers")
        return
      
    #toimage(imgRGB[peak2[0]-10:peak1[0]-5, peak1[1]-30: peak2[1]-25]).show()
    
    # following code assumes that the first is right
    angleAvg = 0
    # above peak 1 should be an edge separating the blue and white surface, this leads to a higher accuracy of the angle
    first = None
    current = None
    blue = matplotlib.colors.rgb_to_hsv(np.array([0.06*2*averageGrayValue,0.24*2*averageGrayValue,0.4*2*averageGrayValue]))
    i = 0

    width = peak2[1]+5-peak1[1]
    #edge = np.zeros( (int(width / 10),2) )
    for c in range(peak1[1]-30, peak2[1]-25, 10):
        if c < peak1[1]-30+width/2:#skip the first half
            pass
        else:
            coloumnHSV = matplotlib.colors.rgb_to_hsv(imgRGB[peak2[0]-10:peak1[0]-5,c])
        
            dSColoumn = []
            for r in range(1,coloumnHSV.shape[0]):#from top to bottom
                dS = coloumnHSV[r][1]-coloumnHSV[r-1][1]
                dSColoumn.append(dS)
                if dS < -0.15 and abs(coloumnHSV[r][0]-blue[0]) < 0.3 and (i == 0 or abs(r-current[0]) <= 5): #edge detection, filter outliers
                    current = (r,c)
                    #edge[i][0] = r
                    #edge[i][1] = c
                    if first is None:
                        first = (r,c)
                        print("first: "+str(first))
                    else:
                        angle = math.degrees(math.atan((abs(current[0]-first[0]))/(abs(current[1]-first[1])))) #using blue/white edge
                        angleAvg += angle
                        print("current: " + str(current) + " angl: "+str(angle)+" i:"+str(i)+"angleAvg:"+str(angleAvg))
                        i += 1
                    break
            #print(dSColoumn)
            #plt.plot(dSColoumn)
            #plt.plot(coloumnHSV[:,2])
    #print(edge)
    
    

    if first is None or current is None:
        print("could not detect blue-white edge")
        
    angleAvg /= i #average
    print("average angle: "+str(angleAvg))
    #angle = math.atan((peak2[0]-peak1[0])/(peak2[1]-peak1[1]))*180/math.pi #using color peaks
    
    #the width of the blue section
    hypo= int(math.sqrt((peak2[0]-peak1[0])*(peak2[0]-peak1[0]) + (peak2[1]-peak1[1])*(peak2[1]-peak1[1])))+30

    return (peak1[0]-110,peak1[0]-35, peak1[1]-10, peak1[1]+hypo, -angleAvg, peak1)


def findMaximumColor(imgHSV, searchRGB, errormarginH, minS):
    '''find peak position in image using scale space'''
    print("calculating histogram to find the color peak pos " +str(searchRGB))
    searchHSV = matplotlib.colors.rgb_to_hsv(searchRGB);
    
    #search for valid pixels (pixels matching search criteria)
    validPixels = np.zeros((imgHSV.shape[0], imgHSV.shape[1]), dtype=bool)
    for y in range(imgHSV.shape[0]):
        for x in range(imgHSV.shape[1]):
            #if abs(scaledHSV[x][y][0]-yellowAreaHSV[0]) < 0.07 and scaledHSV[x][y][1] > 0.3 and scaledHSV[x][y][2]<0.9 and scaledHSV[x][y][2]>0.3:
            if abs(imgHSV[y][x][0]-searchHSV[0]) < errormarginH and imgHSV[y][x][1] > minS and imgHSV[y][x][2] > 0.12:
               validPixels[y][x] = True
               #mark for visualization
               #imgHSV[y][x][0] = 0
               #imgHSV[y][x][1] = 1
               #imgHSV[y][x][2] = 0.5
            #else: 
              #  imgHSV[y][x][2] = 0
    #display(Image.fromarray(np.uint8(validPixels*255)))        

    #calculate histograms of the valid pixels by projecting rows and coloumns

    #rows
    rdir = validPixels.sum(axis=1)
    rdir= gaussian_filter(rdir, sigma=5)

    #coloums  
    cdir = validPixels.sum(axis=0)
    cdir= gaussian_filter(cdir, sigma=5)
        
#    plt.plot(rdir)
#    plt.plot(cdir)
#    plt.xlabel('X-Coordinate')
#    plt.grid(True)
#    plt.show()
         
    if np.sum(rdir)+np.sum(cdir) ==0:
        print("did not find a peak")
        return None
      
    #special rule
    #find low before max, but what defines the low?
    #for r in range(rdir.shape[0]):
    #rdir[x]
       
    rowMax = int(np.argmax(rdir))
    columnMax = int(np.argmax(cdir))
    print("Maimumx r,c:"+str((rowMax, columnMax)))
    return (rowMax, columnMax)
 
def findRectangle(imageRGB):
    '''performs some scans to find the line with the maximum derivative (edges)'''
    imageRGB = scaleSpace(imageRGB,1.2)  # higher precision for more precise results, but more affected by noise
    imageHSV = matplotlib.colors.rgb_to_hsv(imageRGB)
    
    # find line with highest sum of derivative
    maxV = None  # maximum values of global maximum
    for row in range(1,imageRGB.shape[0],10):
        lastV = imageHSV[row][0][2]
        sumderiv = 0
        histo = []
        for coloumn in range(1,imageRGB.shape[1]):
            fx = imageHSV[row][coloumn][2]
            deriv = fx - lastV
            sumderiv += abs(deriv)
            histo.append(deriv)
            lastV = fx

        # found new maximum
        if maxV == None or sumderiv > maxV[0]:
            maxV = (sumderiv, row, histo)  # sum of derivative of V, row, histogram of V


    if maxV == None:
        print("no central line found")
        return
        
    RectTupleClass = namedtuple("Rectangle", "left right top bottom")
    # look for biggest change in this line of Saturation
    horLineHSV = imageHSV[maxV[1],:]
    dS = np.diff(horLineHSV[:,1])
    
    #plt.plot(dS)
    #plt.plot(horLineHSV[:,1])
    #plt.xlabel('X-Coordinate')
    #plt.show()
    
    leftBorder = 0
    rightBorder = len(horLineHSV)
    # scan horizontally
    c= leftBorder
    while c < rightBorder-1:
        if leftBorder==0:  # first occurence is left border
            if dS[c] <= -0.027:  # change in saturation must be big enough
                leftBorder = c + 11  # better if looking for local minimum, currently only adding fixed distortion
                # skip the next coloums
                c += 150
        elif 0.15 < horLineHSV[c][0] < 0.84 and dS[c] >= 0.027 and horLineHSV[c][1] > 0.2:  # not red and is coloful
            print("H: "+str(horLineHSV[c][0]))
            print("dS: "+str(dS[c]))
            print("S: "+str(horLineHSV[c][1]))
            rightBorder= c-3
            break  # first occurrence
        c+=1

    # scan vertically
    verLineHSV = imageHSV[:,int(len(horLineHSV)/2)]  # middle
    dS = np.diff(verLineHSV[:,1])
    ddS = np.diff(dS)
    
    topBorder = 0
    bottomBorder = len(verLineHSV)
    localMinimum = False
    localMaximum = False
    for r in range(topBorder, bottomBorder-2):
        if topBorder == 0:  # first occurence
            if dS[r] <= -0.027:  # change in saturation must be big enough
                localMinimum = True
            if localMinimum and ddS[r] > 0: # wait till it rises again
                topBorder = r
        else:
            if (bottomBorder == len(verLineHSV)  # first occurence
                and 0.15 < verLineHSV[r][0] < 0.84   # not red,
                and dS[r] >= 0.027
            ):
                localMaximum = True
            if localMaximum and ddS[r] < 0:
                bottomBorder = r
    
    #if no result was found, take the last
    if localMaximum and bottomBorder == len(verLineHSV):
        bottomBorder = r
    
    return RectTupleClass(leftBorder,rightBorder, topBorder, bottomBorder)

def rotate2Dvector(point, angle, origin=(0, 0)):
    '''rotates a 2d vector'''
    cos_theta, sin_theta = math.cos(angle), math.sin(angle)
    x, y = point[0] - origin[0], point[1] - origin[1]#translate
    Point = namedtuple('Point', 'x y')
    return Point(x * cos_theta - y * sin_theta + origin[0],
            x * sin_theta + y * cos_theta + origin[1])

def ocr(digitsRGB):
    '''

    performs ocr using tesseract, input is list of numpy images
    todo improve using https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality

    :param digitsRGB:
    :return:
    '''
    import pytesseract
    digitValue = int(1e5) # value of the next digit
    asValue = int(0) #resulting value
    
    for digitRGB in digitsRGB:     
        digitBinary = np.zeros([digitRGB.shape[0],digitRGB.shape[1]])
        for r in range(digitRGB.shape[0]):
            for c in range(digitRGB.shape[1]):
                if digitValue>100:
                    if np.sum(digitRGB[r][c][:]) > 250:#everything bright to white
                        digitBinary[r][c] = 255
                else:
                    digitHSV= matplotlib.colors.rgb_to_hsv(digitRGB)
                    if digitHSV[r][c][1] < 0.23 or abs(0.5-digitHSV[r][c][0]) < 0.4: #everything with low saturation or non red is white
                        digitBinary[r][c] = 255
                #digitBinary[r][c] = np.sum(digitRGB[r][c][:])/3
        #Image.fromarray(np.uint8(digitBinary), 'L').show()
        
        
        #remove black bottom border for better ocr
        foundWhite=False
        bottomBorder=digitRGB.shape[0]-1
        middle = digitRGB.shape[1]//2
        while (not foundWhite):
            if digitBinary[bottomBorder][middle]==0:
                bottomBorder -= 1
            else:
                foundWhite = True
        
        display(Image.fromarray(np.uint8(digitBinary)))
        #toimage(digitBinary).show()    
        
        #todo detect low confidence     
        
        #not working 
        #ocrResult = pytesseract.run_and_get_output(image, 'txt', lang=None, config='-psm 10 -c tessedit_char_whitelist=0123456789', nice=0)
        
        #returns wrong results
        #ocrResult = pytesseract.image_to_data(digitBinary[5:bottomBorder], config='-psm 10 -c tessedit_char_whitelist=0123456789', output_type="dict")
        #print(ocrResult)
        #digitasNumber = ocrResult["text"][0]
        
        digitasNumber = pytesseract.image_to_string(digitBinary[5:bottomBorder], config='-psm 10 -c tessedit_char_whitelist=0123456789')
        
        if digitasNumber == "":
            digitasNumber = 0
        else:
            digitasNumber = int(digitasNumber)
        asValue += int(digitasNumber*digitValue)
        digitValue = digitValue//10#using int for numerical stability
    asValue /= 1000
    print(asValue)
    return asValue
   
def segmentDigits(whiteRect, segmentRGB, draw):
    '''
    segment digits from rectangle
    :param whiteRect:
    :param segmentRGB:
    :param draw:
    :return:
    '''
    digits = []
    numberOfDigits = 8
    width = whiteRect.right - whiteRect.left
    if width < 5:
        print("Width of segment is too small. Segmentation failed.")
    stepSize = int(width / numberOfDigits)
    xLeftBorder = whiteRect.left+stepSize*2
    
    #toimage(segmentRGB[whiteRect.top : whiteRect.bottom, whiteRect.left : whiteRect.right]).show()
    for x in range(whiteRect.left, whiteRect.right, stepSize):
        # draw.line(
#             (cornerSegment.x + x,
#             cornerSegment.y,
#             cornerSegment.x + x,
#             cornerSegment.y + segmentHeight)
#         )
        if x > whiteRect.left+stepSize*2:#skip first line and first two digit
            newDigit = segmentRGB[whiteRect.top : whiteRect.bottom, xLeftBorder+5 : x]#little bit of offset because of the border
            digits.append(newDigit)
            #toimage(newDigit).show()
            xLeftBorder = x
        
    return digits

def save(date,number):
    with open("numbers.txt", "a") as myfile:
        myfile.write(str(date.year)+"-"+str(date.month)+"-"+str(date.day)+"-"+str(date.hour)+"-"+str(date.minute) +" "+str(number)+"\n")
    
def loadLast():
    # with open("numbers.txt", "r") as f:
 #        f.seek(-2, os.SEEK_END)     # Jump to the second last byte.
 #        while f.read(1) != b"\n":   # Until EOL is found...
 #            f.seek(-2, os.SEEK_CUR) # ...jump back the read byte plus one more.
 #        last= f.readline()         # Read last line.
        
    import subprocess
    last = str(subprocess.check_output(['tail', '-1', "numbers.txt"]))[2:]
    
    last_date = []
    dStr = last[:last.rfind('/')]
    dStr = dStr[dStr.rfind('/')+1:]
    last_date.append(int(dStr[:dStr.find('-')]))#year
    dStr = dStr[dStr.find('-')+1:]
    last_date.append(int(dStr[:dStr.find('-')]))#month
    dStr = dStr[dStr.find('-')+1:]
    last_date.append(int(dStr[:dStr.find('-')]))#day
    dStr = dStr[dStr.find('-')+1:]
    last_date.append(int(dStr[:dStr.find('-')]))#hour
    dStr = dStr[dStr.find('-')+1:]
    last_date.append(int(dStr[:dStr.find(' ')]))#minute

    dStr = dStr[dStr.find(' ')+1:-2]
    lastvalidnumber = float(dStr)
    last_date = datetime.datetime(last_date[0],last_date[1],last_date[2],last_date[3],last_date[4]);
    return last_date, lastvalidnumber
    
def whiteBalance(sourceRGB, destRGB):
    #white balance so that the average value is as defined
    averageRGB = [sourceRGB[:, :, i].mean() for i in range(3)]
    print("average RGB "+str(averageRGB))
    
    global averageGrayValue 
    averageGrayValue=127
    wbCorrection = np.divide([averageGrayValue, averageGrayValue, averageGrayValue], averageRGB)#average 0.5 rgb value
    print("WB: "+str(wbCorrection))
    destRGB[:,:] = np.multiply(destRGB[:,:], wbCorrection)
    return destRGB
    
    
def getStringFromImage(path):
    rgbOrigPIL= Image.open(path)
    rgbOrigPIL.load()
    rgbOrig = np.asarray( rgbOrigPIL, dtype="float32" )
    print("Input dimension: "+str(rgbOrig.shape))
    
    # use only a cut of the original pucture
    reductionWidth = 0.48
    reductionHeight = 0.1
    print(str(reductionWidth*100)+"%x"+str(reductionHeight*100)+"% image size reduction")
    rgb = np.array(rgbOrig[int(rgbOrig.shape[0]*reductionHeight/2):int(rgbOrig.shape[0]*(1-reductionHeight/2)), int(rgbOrig.shape[1]*reductionWidth/2):int(rgbOrig.shape[1]*(1-reductionWidth/2)),:], copy=True) 
    print("Scaled dimension: "+str(rgb.shape))
    whiteBalance(rgb)
    
    #correctedRGB= matplotlib.colors.hsv_to_rgb(imgHSV)
    #pilImage = Image.fromarray(rgbOrig, mode='RGB')
    #rgb = np.clip(rgb,0,1)
    #rgb[:,:] = np.multiply(rgbOrig[:,:], [256,256,256])
    #pilImage = Image.fromarray(rgb, mode='RGB')#here only showing shit
    #pilImage.show()
    #toimage(rgb).show()#shows correctly

    imgHSV = matplotlib.colors.rgb_to_hsv(rgb)
    
    # normalize
    minV = np.min(imgHSV[:,:,2])
    imgHSV[:, :, 2] -= minV # minimum value at 0

    maxV = np.max(imgHSV[:,:,2])
    imgHSV[:, :, 2] /= maxV # maximum value is 1
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
    correctedRGB = np.asarray(pilRGB)#pillow to numpy
    
    # rgb = ndimage.interpolation.rotate(correctedRGB, segRange[4], reshape=False)#rotate using scipy ndimage
    correctedRGB = correctedRGB[segRange[0]:segRange[1], segRange[2]:segRange[3], :]
    #toimage(correctedRGB).show()#show segment
    
    #rgb = rgb.reshape(rgb.shape[0]*rgb.shape[1], rgb.shape[2])
    #  translate image by the difference of
    draw = ImageDraw.Draw(rgbOrigPIL)
    # draw image size reduction
    draw.rectangle(
        (rgbOrig.shape[1]*reductionWidth/2,
        rgbOrig.shape[0]*reductionHeight/2,
        rgbOrig.shape[1]*(1-reductionWidth/2),
        rgbOrig.shape[0]*(1-reductionHeight/2)
        ), outline="blue", fill=None)
   
    # rotate point around anchor, bug: reduction is shorter because of rotation, current implementation does not shrink it
    rotationAnchor = (rgbOrig.shape[1]*reductionWidth/2 + rotationAnchor[0], rgbOrig.shape[0]*reductionHeight/2 +rotationAnchor[1]) # from r,c to x,y
    rgbOrigPIL = rgbOrigPIL.rotate(segRange[4], resample=Image.BILINEAR, center=rotationAnchor)
    # top or bottom left corner relative to origninal image
    cornerAbsolute = (rgbOrig.shape[1]*reductionWidth/2 + segRange[2], rgbOrig.shape[0]*reductionHeight/2+ segRange[0])
    cornerSegment = rotate2Dvector(cornerAbsolute, math.radians(segRange[4]), rotationAnchor) # position after rotation?
    
    # draw things
    segmentWidth = segRange[3] - segRange[2]
    segmentHeight= segRange[1] - segRange[0]
    draw = ImageDraw.Draw(rgbOrigPIL)
    
    # draw segmentation
    draw.rectangle(
        (cornerSegment.x,
        cornerSegment.y ,
        cornerSegment.x + segmentWidth,
        cornerSegment.y + segmentHeight
        ), outline="red", fill=None)
    
    whiteRect = findRectangle(correctedRGB) # find rectangle in this segment
    print(whiteRect)
    # draw central line
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

    
    digits = segmentDigits(whiteRect,correctedRGB, draw)
    #show marked picture
    toimage(rgbOrigPIL).show()
    return ocr(digits)


if __name__ == '__main__':
    input = "./images/image.jpg"#default input
    if len(sys.argv) > 1:
        input = sys.argv[1]
      
    date = []  
    dStr = input[:input.rfind('/')]
    dStr = dStr[dStr.rfind('/')+1:]
    date.append(int(dStr[:dStr.find('-')]))#year
    dStr = dStr[dStr.find('-')+1:]
    date.append(int(dStr[:dStr.find('-')]))#month
    dStr = dStr[dStr.find('-')+1:]
    date.append(int(dStr))#day
    date.append(int(input[input.rfind('/')+1:input.rfind('_')]))#hour
    date.append(int(input[input.rfind('_')+1:input.rfind('.')]))#minute
    
    date = datetime.datetime(*date)#date from list into datetime object
    last_date, lastvalidnumber = loadLast()
    if date > last_date:
        newNumber = getStringFromImage(input)
        if newNumber < lastvalidnumber:
            print("OCR returned impossible result. Rejected.")
        else:
            save(date,newNumber)
    else:
        print("input is older then last valid data")
