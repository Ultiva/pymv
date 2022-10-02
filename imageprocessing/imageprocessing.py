import cv2
import numpy as np

__all__ = ['ImageProcessing']

class ImageProcessing():
    #def __init__(self):
    #    pass


    @staticmethod
    def preprocessForRoI(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 3)
        image = cv2.Canny(image, 20, 70)
        kernel = np.ones((3, 3))
        image = cv2.dilate(image, kernel, iterations=1)
        #image = cv2.erode(image, kernel, iterations=1)
        return image


    @staticmethod
    def getBiggestClosedContour(image):
        biggest = np.array([])
        maxArea = 0
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if area > maxArea and len(approx) == 4:
                    biggest = approx
                    maxArea = area
        cv2.drawContours(image, biggest, -1, (255, 0, 0), 20)
        # cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
        return biggest


    def reorder(pts):
        myPoints = pts.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), np.int32)
        add = myPoints.sum(1)
        # print("add", add)
        myPointsNew[0] = pts[np.argmin(add)]
        myPointsNew[3] = pts[np.argmax(add)]
        diff = np.diff(pts, axis=1)
        myPointsNew[1] = pts[np.argmin(diff)]
        myPointsNew[2] = pts[np.argmax(diff)]
        return myPointsNew


    @staticmethod
    def getWarp(img, biggest): # image and biggest contour

        # crop
        widthImg, heightImg = 640, 1280
        biggest = ImageProcessing.rearrange(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
        imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

        return imgCropped





    @staticmethod
    def image_smoothening(image):
        BINARY_THREHOLD = 180
        ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image

    @staticmethod
    def remove_noise_and_smooth(image):
        img = cv2.imread(image, 0)
        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                         3)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = ImageProcessing.image_smoothening(img)
        or_image = cv2.bitwise_or(img, closing)
        return image

    @staticmethod
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    @staticmethod
    def remove_noise(image):
        return cv2.medianBlur(image, 3)

    # thresholding
    @staticmethod
    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # dilation
    @staticmethod
    def dilate(image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    # erosion
    @staticmethod
    def erode(image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    @staticmethod
    def opening(image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def closing(image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # canny edge detection
    @staticmethod
    def canny(image):
        return cv2.Canny(image, 100, 200)

    # skew correction
    @staticmethod
    def deskew(image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated