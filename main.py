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




"""
def loadOCR(ocr, useGPU):
    if ocr == "pytesseract":
        return pytesseract.image_to_string
    elif ocr == "easyocr":
        return easyocr.Reader(['en'], gpu=useGPU)
    else:
        print("OCR Model not supported. Fallback to easyocr")
        return easyocr.Reader(['en'])



def cleanupText(text):
    # strip out non-ASCII text -> draw text on the image
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()
"""




def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgCanny = cv2.Canny(imgBlur, 20, 70)
    kernel = np.ones((3,3))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    #return imgDial
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres






def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(img, biggest, -1, (255, 0, 0), 20)
    #cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

















def captureVideo():
    """
        #OCR
        text = ocr.readtext(warpedImg)
        # loop over recognized text per image
        for (bbox, text, prob) in text:
            print("[INFO] {:.4f}: {}".format(prob, text))
            # bbox
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            # clean up the text and draw the box surrounding the text along
            # with the OCR'd text itself
            text = cleanupText(text)
            cv2.rectangle(warpedImg, tl, br, (0, 255, 0), 2)
            cv2.putText(warpedImg, text, (tl[0], tl[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)



        # barcode
        for barcode in decode(warpedImg):
            barcode_data = barcode.data.decode('utf-8')
            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(warpedImg, [pts], True, (255, 0, 255), 5)
            cv2.putText(warpedImg, barcode_data, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)




        cv2.imshow("warp", warpedImg)





        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    """



def run(
        height=800,
        width=800,
        ocr="easyocr",
        gpu=False,
        magnification=4,
        srmodel="espcn",
):

    bc = Barcode()
    #imgProc = ImageProcessing()
    ocr = OCR(ocr, gpu)
    sr = SuperResolution(srmodel, magnification)

    cap = cv2.VideoCapture(0)
    for i in range(3):
        _, _ = cap.read() # warmup

    # Props list: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    #cap.set(3, args.width)
    #cap.set(4, args.height)


    while True:
        _, frame = cap.read()
        cv2.imshow("frame", frame)

        # find corner of biggest closed rectangle and draw those points
        roiPts = ImageProcessing.biggestClosedContour(frame)
        rearrangedRoiPts = ImageProcessing.rearrangePts(roiPts)
        cv2.drawContours(frame, rearrangedRoiPts, -1, (255, 0, 0), 20)

        # warp image with predefined lengths
        warpedImg = ImageProcessing.getWarp(frame, rearrangedRoiPts, width, height)
        cv2.imshow("warp", warpedImg)

        # Superresolution
        #srImg = cv2.resize(warpedImg, dsize=None, fx=2, fy=2)
        #srImg = sr.upsample(warpedImg)
        #cv2.imshow("sr", srImg)


        # OCR        
        ocrpreproc = ImageProcessing.preprocessForOCR(warpedImg)
        ocrImg = ocr.readtext(ocrpreproc)
        cv2.imshow("ocr", ocrImg)



        # barcode
        bcImg = bc.decode(warpedImg)
        cv2.imshow("warp", bcImg)

        #imgArray = ([frame, warpedImg], [ocrImg, bcImg])
        #cv2.imshow("Result", ImageProcessing.stackImages(0.5, imgArray))




        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def parse_opt():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--strong-sort-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt') # example
    parser.add_argument("-H", "--height", type=int, default=640, help="crop height")
    parser.add_argument("-W", "--width", type=int, default=480, help="crop width")
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