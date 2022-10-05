# later for GUI
import sys
import os
#from PyQt5 import QtWidgets
#from PyQt5.QtWidgets import QApplication, QApplication
#from PySide6.QWidgets import QApplication, QMainWindow


# Img Processing
import cv2
import numpy as np

# Others
#import threading
import argparse
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
#WEIGHTS = ROOT / 'weights'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'barcode') not in sys.path:
    sys.path.append(str(ROOT / 'barcode'))
if str(ROOT / 'imageprocessing') not in sys.path:
    sys.path.append(str(ROOT / 'imageprocessing'))
if str(ROOT / 'ocr') not in sys.path:
    sys.path.append(str(ROOT / 'ocr'))
if str(ROOT / 'superresolution') not in sys.path:
    sys.path.append(str(ROOT / 'superresolution'))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



from barcode import Barcode
from imageprocessing import ImageProcessing
from ocr import OCR
from superresolution import SuperResolution






def run(
        height, # currently predefined for warp
        width, # currently predefined for warp -> later solved by region proposal concept
        ocr="easyocr",
        gpu=False,
        magnification=4,
        srmodel="espcn",
):


    # Object modules
    bc = Barcode()
    ocr = OCR(ocr, gpu)
    sr = SuperResolution(srmodel, magnification)



    # camera
    cap = cv2.VideoCapture(0)
    for i in range(3):
        _, _ = cap.read() # warmup
    # Props list: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    #cap.set(3, 3840)
    #cap.set(4, 2160)


    while True:
        _, frame = cap.read()

        # find corners of biggest closed rectangle, rearrange them and draw those points
        # -> here later on more suitable roi method
        roiPts = ImageProcessing.biggestClosedContour(frame)
        rearrangedRoiPts = ImageProcessing.rearrangePts(roiPts)
        cv2.drawContours(frame, rearrangedRoiPts, -1, (255, 0, 0), 20)


        # warp image with predefined lengths -> see in arguments from run( height and width)
        warpedImg = ImageProcessing.getWarp(frame, rearrangedRoiPts, width, height)
        cv2.imshow("warp", warpedImg)

        # Apply Superresolution on warped image -> more research on this
        #warpedImg = cv2.resize(warpedImg, dsize=None, fx=2, fy=2)
        #warpedImg = sr.upsample(warpedImg)
        #cv2.imshow("sr", srImg)


        # OCR -> currently directly on warped image -> later on establish suitable OCR image preprossesing -> DESCEW, binary,...
        #ocrpreproc = ImageProcessing.preprocessForOCR(frame)
        frame = ocr.readtext(frame)
        #ocrImg = ocr.readtext(warpedImg)
        #cv2.imshow("OCR", ocrImg)


        # barcode
        #bcImg = bc.decode(warpedImg)
        bcImg = bc.decode(frame) # currently in full frame
        cv2.imshow("Barcode", bcImg)



        # imgArray = ([frame, warpedImg], [ocrImg, bcImg])
        #imgArray = ([frame, bcImg])
        #cv2.imshow("Overview", ImageProcessing.stackImages(0.6, imgArray))




        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def parse_opt():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--strong-sort-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt') # example
    parser.add_argument("-H", "--height", type=int, default=960, help="crop height")
    parser.add_argument("-W", "--width", type=int, default=520, help="crop width")
    parser.add_argument("-OCR", "--ocr", type=str, default="easyocr", help="pytesseract, easyocr or keras_ocr")
    parser.add_argument("-GPU", "--gpu", type=bool, default=0, help="Use GPU")
    parser.add_argument("-MAG", "--magnification", type=int, default=2, help="Magnification Factor (2 or 4)")
    parser.add_argument("-SR", "--srmodel", type=str, default="ESPCN", help="Magnification Model (ESPCN, EDSR. FSRCNN, LapSRN, ...)")
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)