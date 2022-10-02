# OCR
import cv2
import pytesseract
import easyocr
#import keras_ocr

__all__ = ['OCR']

class OCR(object):
    def __init__(self, ocrModel, useGPU):
        if ocrModel == "pytesseract":
            self.ocrModel = pytesseract.image_to_string
        elif ocrModel == "easyocr":
            self.ocrModel = easyocr.Reader(['en'], gpu=useGPU)
        else:
            print("OCR Model not supported. Fallback to easyocr")
            self.ocrModel = easyocr.Reader(['en'])


        #self.config = ("-l eng --oem 1 --psm 7")
        #self.api = tesseract.TessBaseAPI()
        #self.api.Init(".", "eng", tesseract.OEM_DEFAULT)
        #self.api.SetPageSegMode(tesseract.PSM_AUTO)






    def readtext(self, img):
        text = self.ocrModel.readtext(img)
        # loop over recognized text per image
        for (bbox, text, prob) in text:
            #print("[INFO] {:.4f}: {}".format(prob, text))
            # bbox
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            # clean up the text and draw the box surrounding the text along
            # with the OCR'd text itself
            text = self.__cleanupText(text)
            cv2.rectangle(img, tl, br, (0, 255, 0), 2)
            cv2.putText(img, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img



    @staticmethod
    def __cleanupText(text):
        # strip out non-ASCII text -> draw text on the image
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()