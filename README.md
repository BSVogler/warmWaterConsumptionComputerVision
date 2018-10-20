# warmWaterConsumptionComputerVision

This repo contains the source code to analyze a warm water counter using python 3. The algorithms are based on the course _Bildverstehen I_ and _Bildverstehen II_ from Carsten Steger (Recommended book: Carsten Steger & Markus Ultrich & Christian Wiedemann - Machine Vision: Algorithms and Applications).

## Current status
Pictures are taken and can be segmented. The results of the segmentation could be more robust.
Currently the used OCR software is returning bad results, so I intend to train some custom machine learning model.

## Recommended Hardware
- Raspberry Pi (32€) + case
- sd card (10€)
- Rasperry Pi Camera (non infrared version, 12€)
- power source cable (recommend a long cable, 9€)
- color printer
- if raspberry pi 2 a wifi dongle (EDIMAX EW-7811UN Wireless USB Adapter, 8€)
- [cables](https://www.amazon.de/gp/product/B0786KBBZ5/ref=oh_aui_detailpage_o02_s02?ie=UTF8&psc=1) (4,29€)
- [white LEDs](https://www.amazon.de/gp/product/B0786KBBZ5/ref=oh_aui_detailpage_o02_s02?ie=UTF8&psc=1) (7€)
- resistors and transistor
- soldering iron

### Hardware installation
It is often easier to change the environment and the hardware than the algorithms and software parameters.
The whole installation should be covered so that the light-conditions remain constant during the day. Therefore a light source is needed. I soldered some white LEDS together to build a flash.

I printed a mask using a high resolution color print.

Installing the light source.

Then I installed them on the side (Auflichtbeleuchtung). When they are installed slightly on the side the depth casts a shadow, which can be used for digit segmentation. In a later version I segment the digits by just detecting the left and right border. Then the light should be installed near the camera. However specular reflections of the transparant plastic surface are not wanted. Only the diffuse reflection should be visible so it is recommended to install them slightly on the side. Alternativly one could use a polarization filter.

For fixating the whole construction you can use cardboard, paper, wood and 3d printed stuff. I just used tape to fixate the whole thing to the wall.

## Costs
You can use a different computer and a different camera. Note that the script for image capturing is inteded to run on a raspberry pi. I got my raspberry pi for free via the career network careerloft.

With working parts: > 74€

## Problems I encountered
The wifi dongle I bought did not run. There are several sold on amazon by chinese merchants under different names. The one I ordered came with a mini [compact disk](https://en.wikipedia.org/wiki/Compact_disc) (who uses compact disks today?) and no download link for the driver. I got a driver from a different merchant for the same product. I was still not able to get the adapter to work.

My camera did not work after I installed it. Removing or adding the camera to the pi while running can probably damage the camera. Either it was already broken or I damaged it. Therefore I bought a second one.

## Segmentation Algorithm
The algorithm works in the following order

- white balance by using RGB average
- look for yellow and blue marks
- calculate angle between the selected marks to correct image rotation
- use color grandient to find left and right border of digits
- scan horizontally for maximum gradient to find central line
- find borders of digit box by using first gradient above a threshold
- split in equally sized parts to obtain digits
- use ocr software "tesseract" on every digit

## Dependencies
numpy, scipy, pytesseract

Install pytesseract with `pip3 --install pytesseract`  

## Installation
Make sure that the dependencies are installed. The script should be run when the raspberry boots. For automatic picture taking I put the following two lines into `/etc/rc.local`
```bash
cd /home/pi
python3 ./takepicture.py >> tlLog.log
```
The segmentation is run afterwards on the collected data. Once the pipeline is working it could be run directly on the pi.
