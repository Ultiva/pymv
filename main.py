import cv2
import time
import PIL
from PIL import Image
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import threading
import numpy as np
import argparse

import torch, torchvision

import pytesseract
import easyocr
from pyzbar.pyzbar import decode
import keras_ocr




parser = argparse.ArgumentParser()
parser.add_argument("-H", "--height", type=int, default=640, help="crop height")
parser.add_argument("-W", "--width", type=int, default=480, help="crop width")
parser.add_argument("-OCR", "--ocr", type=str, default="easyocr", help="pytesseract, easyocr or keras_ocr")
parser.add_argument("-GPU", "--gpu", type=bool, default=0, help="Use GPU")
parser.add_argument("-MAG", "--magnification", type=int, default=2, help="Magnification Factor (2 or 4)")
parser.add_argument("-SR", "--srmodel", type=str, default="ESPCN", help="Magnification Model (ESPCN, ...)")

args = parser.parse_args()


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


def loadSuperRes(srmodel, magnification):
    #https://learnopencv.com/super-resolution-in-opencv/#sec4
    # also try edsr,...

    # currently hardcoded to just ESPCN
    model = srmodel

    if magnification == 2:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(f"./resources/ocr/{model}_x2.pb")
        #sr.readModel(f"./resources/ocr/{model}_x2.pb")
        mag = 2
    elif magnification == 3:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(f"./resources/ocr/{model}_x3.pb")
        mag = 4
    elif magnification == 4:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(f"./resources/ocr/{model}_x4.pb")
        mag = 4
    else:
        print("Magnification Factor not supported. Fallback to 2x")
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(f"./resources/ocr/{model}_x2.pb")
        mag = 2

    sr.setModel(f"{model.lower()}", mag)
    return sr






def captureVideo():
    cap = cv2.VideoCapture(0)
    # Props list: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    #cap.set(3, args.width)
    #cap.set(4, args.height)


    ocr = loadOCR(args.ocr, args.gpu)
    sr = loadSuperRes(args.srmodel, args.magnification)

    while True:
        ret, frame = cap.read()
        cv2.imshow("Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def main():
    captureVideo()




if __name__ == "__main__":
    main()