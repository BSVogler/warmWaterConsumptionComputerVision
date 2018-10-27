#!python3

import datetime
import math
import sys
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import scipy.ndimage as ndimage
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter1d

verbose = False
digitCounter = 0


def findMaximumColor(imgHSV, searchRGB, errormarginH, minS):
    """
    find peak position in image using scale space
    :param imgHSV: numpy image
    :param searchRGB: tuple
    :param errormarginH:
    :param minS:
    :return:
    """

    if verbose:
        print("calculating histogram to find the color peak pos " + str(searchRGB))
    searchH = matplotlib.colors.rgb_to_hsv(searchRGB)[0]

    imgH = imgHSV[:, :, 0]
    imgS = imgHSV[:, :, 1]
    # search for valid pixels (pixels matching search criteria)
    validPixels = (abs(imgH - searchH) < errormarginH).astype(bool) & (imgS > minS).astype(bool)

    # calculate histograms of the valid pixels by projecting rows and coloumns
    # rows
    rdir = validPixels.sum(axis=1)
    rdir = gaussian_filter1d(rdir, sigma=8)

    # coloums
    cdir = validPixels.sum(axis=0)
    cdir = gaussian_filter1d(cdir, sigma=8)

    if verbose:
        plt.plot(rdir)
        plt.plot(cdir)
        plt.xlabel('X/Y-Coordinate')
        plt.grid(True)
        plt.show()

    # special rule
    # find low before max, but what defines the low?
    # for r in range(rdir.shape[0]):
    # rdir[x]

    rowMax = int(np.argmax(rdir))
    columnMax = int(np.argmax(cdir))
    if verbose:
        print("Maimumx r,c:" + str((rowMax, columnMax)))
    return (rowMax, columnMax)


def segment_core(hsv):
    """

    :param hsv:
    :return: coordinates of intersting area and rotated rgb image
    """
    yellowCorrection = (0, 0)
    blueCorrection = (0, 0)
    yellowCenter = findMaximumColor(hsv, (214, 163, 33), errormarginH=0.04, minS=0.3)  # 838ms! per call, intel 806
    yellowCenter = (yellowCenter[0] + yellowCorrection[0], yellowCenter[1] + yellowCorrection[1])
    blueCenter = findMaximumColor(hsv, (12, 41, 94), errormarginH=0.03, minS=0.6)
    blueCenter = [blueCenter[0] + blueCorrection[0], blueCenter[1] + blueCorrection[1]]

    angle = -math.degrees(math.atan((abs(yellowCenter[0] - blueCenter[0])) / (abs(yellowCenter[1] - blueCenter[1]))))
    if verbose:
        print("Rotate by " + str(angle) + "°")

    # blueCenter[0]=yellowCenter[0] # after rotation they have the same row
    # %% apply rotation
    rotatedRGB = Image.fromarray(np.uint8(rgb))  # 0-1 numpy array to uint8 pillow

    draw = ImageDraw.Draw(rotatedRGB)
    draw.rectangle((yellowCenter[1], yellowCenter[0], yellowCenter[1] + 2, yellowCenter[0] + 2), outline="red",
                   fill=None)
    draw.rectangle((blueCenter[1], blueCenter[0], blueCenter[1] + 2, blueCenter[0] + 2), outline="blue", fill=None)

    rotatedRGB = rotatedRGB.rotate(angle, resample=Image.BILINEAR, center=yellowCenter)

    def rotate2Dvector(point, angleRad, origin=(0, 0)):
        """rotates a 2d vector"""
        cos_theta, sin_theta = math.cos(angleRad), math.sin(angleRad)
        x, y = point[0] - origin[0], point[1] - origin[1]  # translate
        Point = namedtuple('Point', 'x y')
        return Point(x * cos_theta - y * sin_theta + origin[0],
                     x * sin_theta + y * cos_theta + origin[1])

    blueCenterRotated = rotate2Dvector(blueCenter, math.radians(angle), yellowCenter)
    blueCenterRotated = [int(blueCenterRotated[0]), int(blueCenterRotated[1])]
    draw = ImageDraw.Draw(rotatedRGB)
    # highlight centers

    draw.rectangle((yellowCenter[1], yellowCenter[0], yellowCenter[1] + 2, yellowCenter[0] + 2), outline="red",
                   fill=None)
    draw.rectangle((blueCenterRotated[1], blueCenterRotated[0], blueCenterRotated[1] + 2, blueCenterRotated[0] + 2),
                   outline="blue", fill=None)

    if verbose:
        display(rotatedRGB)

    rotatedHSV = matplotlib.colors.rgb_to_hsv(rotatedRGB)
    rotatedHSV[:, :, 2] /= 255

    # %% find the interesting segment and fetch the digits
    green = matplotlib.colors.rgb_to_hsv((0.35, 0.4, 0.12))[0]
    # average values orthogonal to interesting line
    averageWidth = 28
    averageHue = rotatedHSV[yellowCenter[0] - int(averageWidth / 2): yellowCenter[0] + int(averageWidth / 2),
                 yellowCenter[1] + 13:yellowCenter[1] + 93,
                 :]
    # display(Image.fromarray(np.uint8(averageHue*255)))
    averageHue = np.sum(averageHue, axis=0) / averageWidth  # average pixels
    # look for left green border
    distanceHue = []
    for offc in range(0, 80):
        c = yellowCenter[1] + offc
        if averageHue[offc][1] > 0.22 and averageHue[offc][2] > 0.4:
            distanceHue.append(abs(averageHue[offc][0] - green))
        else:
            distanceHue.append(float('inf'))

    #    plt.plot(averageHue[:,:])
    #    plt.plot(distanceHue)
    #    plt.xlabel("Value")
    #    plt.ylabel("Distance")
    #    plt.show()

    left = yellowCenter[1] + 13 + np.argmin(distanceHue)

    # right side
    rangeLeft = -80
    averageHue = rotatedHSV[blueCenterRotated[0] - int(averageWidth / 2): blueCenterRotated[0] + int(averageWidth / 2),
                 blueCenterRotated[1] + rangeLeft: blueCenterRotated[1],
                 :]

    # display(Image.fromarray(np.uint8(averageHue*255)))
    averageHue = np.sum(averageHue, axis=0) / averageWidth  # average pixels
    # plt.plot(averageHue[:,:])
    # plt.xlabel("Value")
    # plt.ylabel("Distance")
    # plt.show()

    r = blueCenterRotated[0]
    distanceHue = []

    for offc in range(rangeLeft, 0):
        c = blueCenterRotated[1] + offc
        if averageHue[offc][1] > 0.4 and averageHue[offc][2] > 0.35:
            distanceHue.append(abs(averageHue[offc][0] - green))
        else:
            distanceHue.append(float('inf'))

    if verbose:
        plt.plot(averageHue[:, 0])
        plt.plot(distanceHue)
        plt.xlabel("Pixel X")
        plt.ylabel("Hue")
        plt.show()

        plt.plot(averageHue[:, 1])
        plt.xlabel("Pixel X")
        plt.ylabel("Saturation")
        plt.show()

        plt.plot(averageHue[:, 2])
        plt.xlabel("Pixel X")
        plt.ylabel("Value")
        plt.show()

    right = blueCenterRotated[1] + np.argmin(distanceHue) + rangeLeft

    RectTupleClass = namedtuple("Rectangle", "left right top bottom")
    interestingSegment = RectTupleClass(left, right, yellowCenter[0] - 20, yellowCenter[0] + 30)
    if verbose:
        draw = ImageDraw.Draw(rotatedRGB)
        draw.rectangle(
            (interestingSegment.left - 1,
             interestingSegment.top - 1,
             interestingSegment.right + 1,
             interestingSegment.bottom + 1), outline="green", fill=None
        )

        display(rotatedRGB)

    if (interestingSegment.right - interestingSegment.left) < 40:
        print("Width of segment is too small. Segmentation failed."+str(interestingSegment))
        return None, rotatedRGB

    return interestingSegment, rotatedRGB


def digitsegments_from_segment(rect, segmentRGB):
    """
    segment digits from rectangle
    :param rect: rectangle tuple with left right, top and bottom
    :param segmentRGB:
    :return:
    """
    digits = []
    numberOfDigits = 8
    width = rect.right - rect.left
    digitWidth = int(width / numberOfDigits)
    xLeftBorder = rect.left + digitWidth * 2  # skip first wo digits

    # toimage(segmentRGB[rect.top : rect.bottom, rect.left : rect.right]).show()
    for x in range(rect.left, rect.right, digitWidth):
        if x > rect.left + digitWidth * 2:  # skip first line and first two digit
            digit = segmentRGB[rect.top: rect.bottom,
                    xLeftBorder + 13: x]  # little bit of offset because of the border
            digits.append(digit)
            xLeftBorder = x

    return digits


def digitsegments_from_image(hsv, save_folder=None):
    """

    :param hsv:
    :return:
    """

    # %% show saturation distribution
    # plt.hist(hsv[:,:,1].flatten(), bins=256)
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.show()
    # %% Calcualte scale space
    # returns nd-image
    def scaleSpace(image, sigma):
        # scale space
        # rgbScaleSpace=np.zeros_like(image)
        if verbose:
            print("Scale space transformation")
        rgbScaleSpace = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0), order=0)

        return rgbScaleSpace

    def whiteBalance(sourceRGB, destRGB):
        # white balance so that the average value is as defined
        averageRGB = [sourceRGB[:, :, i].mean() for i in range(3)]
        if verbose:
            print("average RGB " + str(averageRGB))

        global averageGrayValue
        averageGrayValue = 127
        wbCorrection = np.divide([averageGrayValue, averageGrayValue, averageGrayValue],
                                 averageRGB)  # average 0.5 rgb value
        if verbose:
            print("WB: " + str(wbCorrection))
        destRGB[:, :] = np.multiply(destRGB[:, :], wbCorrection)
        return destRGB

    # whiteBalance(sourceRGB=rgb, destRGB=saturated)
    # scaleSpaceImageHSV = scaleSpaceImageHSV[:, boundaries[2]:boundaries[3], :]

    # hsv = hsv[:, boundaries[2]:boundaries[3], :]
    # rgb = rgb[:, boundaries[2]:boundaries[3], :]

    # %% find rightmost yellow and blue pixel

    interesting_segment, rgb = segment_core(hsv)

    if interesting_segment is not None:
        digits = digitsegments_from_segment(interesting_segment, np.asarray(rgb, dtype="float32"))
        if True:
            global digitCounter
            for d in digits:
                if save is not None:
                    Image.fromarray(np.uint8(d)).save(save_folder + "/" + str(digitCounter) + ".jpg")
                    digitCounter += 1
                else:
                    plt.imshow(np.uint8(d), interpolation='nearest')
                    plt.show()
        return digits


# %%
def ocr(digitsRGB):
    """
    performs ocr using tesseract, input is list of numpy images
    todo improve using https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality

    :param digitsRGB:
    :return:
    """
    digitValue = int(1e5)  # value of the next digit
    asValue = int(0)  # resulting value

    for digitRGB in digitsRGB:
        #        plt.hist(digitRGB[:,:,2],bins=256)
        #        plt.show()
        # TODO kontrastnormierung vornehmen
        # kumulatives histogram, grauwerte unter 5% und über 95% ignorieren VGL MVTec vorlesung Folie 84
        digitBinary = np.zeros([digitRGB.shape[0], digitRGB.shape[1]])
        for r in range(digitRGB.shape[0]):
            for c in range(digitRGB.shape[1]):
                if digitValue > 100:
                    if np.sum(digitRGB[r][c][:]) > 128 * 3:  # everything bright to white
                        digitBinary[r][c] = 255
                else:
                    digitHSV = matplotlib.colors.rgb_to_hsv(digitRGB)
                    if digitHSV[r][c][1] < 0.5 or abs(
                            0.5 - digitHSV[r][c][0]) < 0.4:  # everything with low saturation or non red is white
                        digitBinary[r][c] = 255
                # digitBinary[r][c] = np.sum(digitRGB[r][c][:])/3
        # Image.fromarray(np.uint8(digitBinary), 'L').show()

        # remove black bottom border for better ocr
        foundWhite = False
        bottomBorder = digitRGB.shape[0] - 1
        middle = digitRGB.shape[1] // 2
        while (not foundWhite):
            if digitBinary[bottomBorder][middle] == 0:
                bottomBorder -= 1
            else:
                foundWhite = True

        if verbose:
            display(Image.fromarray(np.uint8(digitBinary)))
        # toimage(digitBinary).show()

        # you need to change the line in tesseract/3.05.02/share/tessdata/configs/tsv to "tessedit_pageseg_mode 10"
        # digitasNumber = pytesseract.image_to_data(digitBinary[0:bottomBorder], config='-psm 10 -c tessedit_char_whitelist=0123456789', output_type="dict")
        digitasNumber = pytesseract.image_to_string(digitBinary[0:bottomBorder],
                                                    config='-psm 10 --oem 0 -c tessedit_char_whitelist=0123456789')
        # print(digitasNumber)
        # confidence = digitasNumber['conf'][-1]
        # digitasNumber = digitasNumber['text'][-1]
        confidence = 100
        if confidence < 30:
            print("confidence of digit with value " + str(digitValue) + " too low")
            digitasNumber = 0
        else:
            if digitasNumber == "":
                digitasNumber = 0
            else:
                digitasNumber = int(digitasNumber)
        asValue += int(digitasNumber * digitValue)
        digitValue = digitValue // 10  # using int for numerical stability
    asValue /= 1000
    return asValue


# %%
def meter_from_image(rgb):
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    return ocr(digitsegments_from_image(hsv))


def loadLast():
    # with open("numbers.txt", "r") as f:
    #        f.seek(-2, os.SEEK_END)     # Jump to the second last byte.
    #        while f.read(1) != b"\n":   # Until EOL is found...
    #            f.seek(-2, os.SEEK_CUR) # ...jump back the read byte plus one more.
    #        last= f.readline()         # Read last line.
    # %%
    import subprocess
    last = str(subprocess.check_output(['tail', '-1', "numbers.csv"]))[2:]
    last_date = []
    dStr = last[:last.rfind('/')]
    dStr = dStr[dStr.rfind('/') + 1:]
    last_date.append(int(dStr[:dStr.find(',')]))  # year
    dStr = dStr[dStr.find(',') + 1:]
    last_date.append(int(dStr[:dStr.find(',')]))  # month
    dStr = dStr[dStr.find(',') + 1:]
    last_date.append(int(dStr[:dStr.find(',')]))  # day
    dStr = dStr[dStr.find(',') + 1:]
    last_date.append(int(dStr[:dStr.find(',')]))  # hour
    dStr = dStr[dStr.find(',') + 1:]
    last_date.append(int(dStr[:dStr.find(',')]))  # minute
    dStr = dStr[dStr.find(',') + 1:]

    dStr = dStr[:-2]
    lastvalidMeter = float(dStr)
    last_date = datetime.datetime(last_date[0], last_date[1], last_date[2], last_date[3], last_date[4]);
    return last_date, lastvalidMeter


def save(date, number):
    with open("numbers.csv", "a") as myfile:
        myfile.write(str(date.year) + "," + str(date.month) + "," + str(date.day) + "," + str(date.hour) + "," + str(
            date.minute) + "," + str(number) + "\n")


def processFile(filepath):
    """
    extract text from image and write it to list
    :param filepath: path of image
    """
    global lastvalidMeter
    global last_date

    # get date from path
    date = []
    dStr = filepath[:filepath.rfind('/')]
    dStr = dStr[dStr.rfind('/') + 1:]
    date.append(int(dStr[:dStr.find('-')]))  # year
    dStr = dStr[dStr.find('-') + 1:]
    date.append(int(dStr[:dStr.find('-')]))  # month
    dStr = dStr[dStr.find('-') + 1:]
    date.append(int(dStr))  # day
    date.append(int(filepath[filepath.rfind('/') + 1: filepath.rfind('_')]))  # hour
    date.append(int(filepath[filepath.rfind('_') + 1: filepath.rfind('.')]))  # minute

    date = datetime.datetime(*date)  # date from list into datetime object
    if True:
        # if date > last_date:
        newNumber = meter_from_image(filepath)
        print(str(date) + ": " + str(newNumber))
        if newNumber < lastvalidMeter:
            print("OCR returned impossible result. Rejected.")
        else:
            save(date, newNumber)
            lastvalidMeter = newNumber
            last_date = date
    else:
        print("filepath " + filepath + " is older then last valid data")


# %%
if __name__ == '__main__':
    # input is either a directory or a single image
    if len(sys.argv) > 1:
        path = sys.argv[1]

    last_date, lastvalidMeter = loadLast()
    print(last_date)
    print(lastvalidMeter)

    import os

    # if is directory for every image
    if os.path.isdir(path):
        for directory, subdirectories, files in os.walk(path):
            for file in files:
                filepath = os.path.join(directory, file)
                if ".jpg" in filepath:
                    print(filepath)
                    rgbOrigPIL = Image.open(filepath)
                    rgbOrigPIL.load()
                    rgb = np.asarray(rgbOrigPIL, dtype="float32")[150:-300, 250:-310, :]
                    hsv = matplotlib.colors.rgb_to_hsv(rgb)
                    digitsegments_from_image(hsv, save_folder="digitsDataset")
    else:
        rgbOrigPIL = Image.open(path)
        rgbOrigPIL.load()
        rgb = np.asarray(rgbOrigPIL, dtype="float32")[150:-300, 250:-310, :]
        hsv = matplotlib.colors.rgb_to_hsv(rgb)
        digitsegments_from_image(hsv, save_folder="digitsDataset")
