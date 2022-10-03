from pyzbar.pyzbar import decode
import cv2

__all__ = ['Barcode']


class Barcode(object):
    def __init__(self):
        pass

    def decode(self, image):
        detectedBarcodes = decode(image)
        for barcode in detectedBarcodes:
            (x, y, w, h) = barcode.rect
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            image = cv2.putText(image, barcode.data.decode("utf-8"), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            image = cv2.putText(image, barcode.type, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return image