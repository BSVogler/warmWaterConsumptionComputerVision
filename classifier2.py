#!python3

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from collections import namedtuple
import datetime
import scipy.ndimage as ndimage
from scipy.ndimage.filters import gaussian_filter

def getMeterFromImage(filename):
    #%%
    yellowCorrection = (0,0)
    blueCorrection = (0,0)
    rgbOrigPIL= Image.open(path)
    rgbOrigPIL.load()
    rgb = np.asarray(rgbOrigPIL, dtype="float32")[100:-300,200:-380,:]
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    
    #%% show saturation distribution
    #plt.hist(hsv[:,:,1].flatten(), bins=256)
    #plt.xlabel("Value")
    #plt.ylabel("Frequency")
    #plt.show()
    #%% Calcualte scale space
    #returns nd-image
    def scaleSpace(image, sigma):
        #scale space
        #rgbScaleSpace=np.zeros_like(image)
        print("Scale space transformation")
        rgbScaleSpace = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0), order=0)
    
        return rgbScaleSpace
    scaleSpaceImageHSV = matplotlib.colors.rgb_to_hsv(scaleSpace(rgb,2)) #53ms per call, intel 49ms
    
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
    
    whiteBalance(sourceRGB=rgb, destRGB=saturated)
    #scaleSpaceImageHSV = scaleSpaceImageHSV[:, boundaries[2]:boundaries[3], :]
    
    display(Image.fromarray(np.uint8(saturated)))
    hsv = hsv[:, boundaries[2]:boundaries[3], :]
    rgb = rgb[:, boundaries[2]:boundaries[3], :]
    
    #%% find rightmost yellow and blue pixel
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
    
    #yellow = matplotlib.colors.rgb_to_hsv((127,127,0))
    saturatedHSV = matplotlib.colors.rgb_to_hsv(saturated)
    #saturatedHSV = matplotlib.colors.rgb_to_hsv(cf.scaleSpace(matplotlib.colors.hsv_to_rgb(saturatedHSV),2))
    #display(Image.fromarray(np.uint8(matplotlib.colors.hsv_to_rgb(saturatedHSV)))) 
    yellowCenter = findMaximumColor(saturatedHSV,(153,197,0), errormarginH=0.02, minS=0.3) #838ms! per call, intel 806
    yellowCenter = (yellowCenter[0]+yellowCorrection[0], yellowCenter[1]+yellowCorrection[1])
    blueCenter = findMaximumColor(saturatedHSV,(58,135,165), errormarginH=0.2, minS=0.2)
    blueCenter = [blueCenter[0]+blueCorrection[0], blueCenter[1]+blueCorrection[1]]
    
    #foundyellow = False
    #for r in range(hsv.shape[0]):
    #    for c in range(hsv.shape[1]):
    #        if foundyellow and abs(hsv[r,c,0]-yellow[0]) < 0.1:
                
    #        foundyellow=True
            
    angle = -math.degrees(math.atan((abs(yellowCenter[0]-blueCenter[0]))/(abs(yellowCenter[1]-blueCenter[1]))))
    print("Rotate by "+str(angle)+"°")
    
    #blueCenter[0]=yellowCenter[0] # after rotation they have the same row
    #%% apply rotation
    rotatedRGB = Image.fromarray(np.uint8(rgb))#0-1 numpy array to uint8 pillow
    
    draw = ImageDraw.Draw(rotatedRGB)
    draw.rectangle((yellowCenter[1], yellowCenter[0], yellowCenter[1]+2, yellowCenter[0]+2), outline="red", fill=None)
    draw.rectangle((blueCenter[1], blueCenter[0], blueCenter[1]+2, blueCenter[0]+2), outline="blue", fill=None)
    
    rotatedRGB = rotatedRGB.rotate(angle, resample=Image.BILINEAR, center=yellowCenter)
    
    def rotate2Dvector(point, angleRad, origin=(0, 0)):
        '''rotates a 2d vector'''
        cos_theta, sin_theta = math.cos(angleRad), math.sin(angleRad)
        x, y = point[0] - origin[0], point[1] - origin[1]#translate
        Point = namedtuple('Point', 'x y')
        return Point(x * cos_theta - y * sin_theta + origin[0],
                x * sin_theta + y * cos_theta + origin[1])
        
    blueCenterRotated = rotate2Dvector(blueCenter, math.radians(angle), yellowCenter)
    blueCenterRotated = [int(blueCenterRotated[0]),int(blueCenterRotated[1])]
    print(blueCenter)
    draw = ImageDraw.Draw(rotatedRGB)
    #highlight centers
    
    draw.rectangle((yellowCenter[1], yellowCenter[0], yellowCenter[1]+2, yellowCenter[0]+2), outline="red", fill=None)
    draw.rectangle((blueCenterRotated[1], blueCenterRotated[0], blueCenterRotated[1]+2, blueCenterRotated[0]+2), outline="blue", fill=None)
    
    display(rotatedRGB)
    
    rotatedHSV = matplotlib.colors.rgb_to_hsv(rotatedRGB)
    rotatedHSV[:,:,2] /= 255
    
    #%% find the interesting segment and fetch the digits
    green = matplotlib.colors.rgb_to_hsv((0.56,0.64,0.36))[0]
    #average values orthogonal to interesting line
    averageWidth = 28
    averageHue = rotatedHSV[yellowCenter[0]-int(averageWidth/2) : yellowCenter[0]+int(averageWidth/2),
                          yellowCenter[1]+40:yellowCenter[1]+120,
                          :]
    #display(Image.fromarray(np.uint8(averageHue*255)))
    averageHue = np.sum(averageHue, axis=0)/averageWidth #average pixels
    #look for left green border
    distanceHue = []
    for offc in range(0, 80):
        c = yellowCenter[1]+offc
        if averageHue[offc][1]>0.3 and averageHue[offc][2]>0.3:
            distanceHue.append(abs(averageHue[offc]-green))
    
    #plt.plot(averageHue[:,:])
    #plt.plot(distanceHue)
    #plt.xlabel("Value")
    #plt.ylabel("Distance")
    #plt.show()
    
    left = yellowCenter[1]+40+np.argmin(distanceHue)
    
    #right side
    rangeLeft = -80
    averageHue = rotatedHSV[blueCenterRotated[0]-int(averageWidth/2) : blueCenterRotated[0]+int(averageWidth/2),
                          blueCenterRotated[1]+rangeLeft : blueCenterRotated[1],
                          :]
    
    #display(Image.fromarray(np.uint8(averageHue*255)))
    averageHue = np.sum(averageHue, axis=0)/averageWidth #average pixels
    #plt.plot(averageHue[:,:])
    #plt.xlabel("Value")
    #plt.ylabel("Distance")
    #plt.show()
    
    r = blueCenterRotated[0]
    distanceHue = []
    
    for offc in range(rangeLeft, 0):
        c = blueCenterRotated[1]+offc
        if averageHue[offc][1]>0.2 and averageHue[offc][2]>0.5:
            distanceHue.append(abs(averageHue[offc][0]-green))
        else:
            distanceHue.append(float('inf'))
        
    #plt.plot(distanceHue)
    #plt.plot(averageHue)
    #plt.xlabel("Value")
    #plt.ylabel("Distance")
    #plt.show()
    right = blueCenterRotated[1]+np.argmin(distanceHue)+rangeLeft
    
    RectTupleClass = namedtuple("Rectangle", "left right top bottom")
    interestingSegment = RectTupleClass(left, right, yellowCenter[0]-15, yellowCenter[0]+33)
    draw = ImageDraw.Draw(rotatedRGB)
    draw.rectangle(
            (interestingSegment.left-1,
            interestingSegment.top-1,
            interestingSegment.right+1,
            interestingSegment.bottom+1), outline="green", fill=None
        )
    display(rotatedRGB)
    
    #%%
    import pytesseract
    def ocr(digitsRGB):
        '''
    
        performs ocr using tesseract, input is list of numpy images
        todo improve using https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality
    
        :param digitsRGB:
        :return:
        '''
        digitValue = int(1e5) # value of the next digit
        asValue = int(0) #resulting value
    
        for digitRGB in digitsRGB:
    #        plt.hist(digitRGB[:,:,2],bins=256)
    #        plt.show()
            #TODO kontrastnormierung vornehmen
            #kumulatives histogram, grauwerte unter 5% und über 95% ignorieren VGL MVTec vorlesung Folie 84
            digitBinary = np.zeros([digitRGB.shape[0],digitRGB.shape[1]])
            for r in range(digitRGB.shape[0]):
                for c in range(digitRGB.shape[1]):
                    if digitValue > 100:
                        if np.sum(digitRGB[r][c][:]) > 128*3:#everything bright to white
                            digitBinary[r][c] = 255
                    else:
                        digitHSV= matplotlib.colors.rgb_to_hsv(digitRGB)
                        if digitHSV[r][c][1] < 0.27 or abs(0.5-digitHSV[r][c][0]) < 0.4: #everything with low saturation or non red is white
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
    

            #you need to change the line in tesseract/3.05.02/share/tessdata/configs/tsv to "tessedit_pageseg_mode 10"
            digitasNumber = pytesseract.image_to_data(digitBinary[5:bottomBorder], config='-psm 10 -c tessedit_char_whitelist=0123456789', output_type="dict")
            if digitasNumber['conf'][4] < 60:
                print("confidence of digit with value "+str(digitValue)+" too low")
                digitasNumber = 0
            else:
                digitasNumber = int(digitasNumber['text'][4])
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
        digitWidth = int(width / numberOfDigits)
        xLeftBorder = whiteRect.left+digitWidth*2 #skip first wo digits
    
        #toimage(segmentRGB[whiteRect.top : whiteRect.bottom, whiteRect.left : whiteRect.right]).show()
        for x in range(whiteRect.left, whiteRect.right, digitWidth):
            if x > whiteRect.left+digitWidth*2:#skip first line and first two digit
                digit = segmentRGB[whiteRect.top : whiteRect.bottom, xLeftBorder+13 : x]#little bit of offset because of the border
                digits.append(digit)
                #toimage(digit).show()
                xLeftBorder = x
    
        return digits
    
    digits = segmentDigits(interestingSegment,np.asarray(rotatedRGB, dtype="float32"), draw)
    for d in digits:
        display(Image.fromarray(np.uint8(d)))
    return ocr(digits)

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
    last_date.append(int(dStr[:dStr.find(',')]))#year
    dStr = dStr[dStr.find('-')+1:]
    last_date.append(int(dStr[:dStr.find(',')]))#month
    dStr = dStr[dStr.find('-')+1:]
    last_date.append(int(dStr[:dStr.find(',')]))#day
    dStr = dStr[dStr.find('-')+1:]
    last_date.append(int(dStr[:dStr.find(',')]))#hour
    dStr = dStr[dStr.find('-')+1:]
    last_date.append(int(dStr[:dStr.find(',')]))#minute

    dStr = dStr[dStr.find(',')+1:-2]
    lastvalidnumber = float(dStr)
    last_date = datetime.datetime(last_date[0],last_date[1],last_date[2],last_date[3],last_date[4]);
    return last_date, lastvalidnumber

def save(date,number):
    with open("numbers.csv", "a") as myfile:
        myfile.write(str(date.year)+","+str(date.month)+","+str(date.day)+","+str(date.hour)+","+str(date.minute) +","+str(number)+"\n")

#%%
if __name__ == '__main__':
    path = "./images/2018-07-14/18_34.jpg"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    #get date from path
    date = []
    dStr = path[:path.rfind('/')]
    print(dStr)
    dStr = dStr[dStr.rfind('/')+1:]
    print(dStr)
    date.append(int(dStr[:dStr.find('-')]))#year
    dStr = dStr[dStr.find('-')+1:]
    date.append(int(dStr[:dStr.find('-')]))#month
    dStr = dStr[dStr.find('-')+1:]
    date.append(int(dStr))#day
    date.append(int(path[path.rfind('/')+1 : path.rfind('_')]))#hour
    date.append(int(path[path.rfind('_')+1 : path.rfind('.')]))#minute
    
    date = datetime.datetime(*date)#date from list into datetime object
    last_date, lastvalidnumber = loadLast()
    if date > last_date:
        newNumber = getMeterFromImage(path)
        if newNumber < lastvalidnumber:
            print("OCR returned impossible result. Rejected.")
        else:
            save(date,newNumber)
    else:
        print("input is older then last valid data")
