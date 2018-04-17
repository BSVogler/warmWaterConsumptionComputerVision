# warmWaterConsumptionComputerVision

This repo contains the source code to analyze a warm water counter using python 3.
Currently the OCR is returning bad results and the segmentation of the letters it not very robust.

## Recommended Hardware
- Raspberry Pi (32€) + case
- sd card (10€)
- Rasperry Pi Camera (non infrared version, 12€)
- power source cable (recommend a long cable, 9€)
- if raspberry pi 2 a wifi dongle (EDIMAX EW-7811UN Wireless USB Adapter, 8€)
- [cables](https://www.amazon.de/gp/product/B0786KBBZ5/ref=oh_aui_detailpage_o02_s02?ie=UTF8&psc=1) (4,29€)
- [white LEDs](https://www.amazon.de/gp/product/B0786KBBZ5/ref=oh_aui_detailpage_o02_s02?ie=UTF8&psc=1) (7€)
- resistors and transistor
- soldering iron

## Cost
You can use a different computer and a different camera. Note that the script for image capturing is inteded to run on a raspberry pi. I got my raspberry pi for free via the career network careerloft.

With working parts: > 74€

## Problems I encountered
The wifi dongle I bought did not run. There are several sold on amazon by chinese merchants under different names. The one I ordered came with a little disk (who uses those little disks today?) and no download link for the driver. I got a driver from a different merchant for the same product. I was still not able to get the adapter to work.

My camera did not work after I installed it. Removing or adding the camera to the pi while running can probably damage the camera. Either it was already broken or I damaged it. Therefore I bought a second one.

## Segmentation Algorithm
The algorithm works in the following order

- white balance by using RGB average
- look for yellow and blue spot
- calculate angle between the selected spots to correct image rotation
- improve accuracy of angle by tracking white/blue edge
- take segment some pixels above
- scan horizontally for maximum gradient to find central line
- find borders of digit box by using first gradient above a threshold
- split in equally sized parts to obtain digits
- use ocr software "tesseract" on every digit

##Dependencies
numpy, scipy, pytesseract

Install with `pip3 --install pytesseract`  
