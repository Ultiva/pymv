import cv2
import numpy as np

__all__ = ['ImageProcessing']

class ImageProcessing():
    #def __init__(self):
    #    pass

    @staticmethod
    def stackImages(scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y],
                                                    (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale,
                                                    scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y],
                                                                                     cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                             scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver



    @staticmethod
    def preprocessForRoI(image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=3, sigmaY=0)
        image = cv2.GaussianBlur(image, (3, 3), 1)
        image = cv2.Canny(image, 20, 70)
        kernel = np.ones((3, 3))
        image = cv2.dilate(image, kernel, iterations=1)
        #image = cv2.erode(image, kernel, iterations=1)
        return image



    @staticmethod
    def preprocessForOCR(image: np.ndarray) -> np.ndarray:
        #norm_img = np.zeros((image.shape[0], image.shape[1]))
        #img = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
        #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(image, kernel, iterations=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        return img


    @staticmethod
    def biggestClosedContour(image: np.ndarray) -> np.ndarray:
        biggestContour = np.zeros([4, 1, 2], dtype=int)
        maxArea = 0
        image = ImageProcessing.preprocessForRoI(image)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if area > maxArea and len(approx) == 4: # rectangle
                    biggestContour = approx
                    maxArea = area
        #cv2.drawContours(image, biggest, -1, (255, 0, 0), 20)
        # cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)

        return biggestContour # Points of biggest rectangular contour


    # convention: tl: 0; tr: 1; bl: 2; br: 3
    def rearrangePts(currPts): #
        pts = currPts.reshape((4, 2))
        rearrangedPts = np.zeros((4, 1, 2), np.int32)

        add = pts.sum(1)
        diff = np.diff(pts, axis=1)

        rearrangedPts[0] = pts[np.argmin(add)]  # tl
        rearrangedPts[3] = pts[np.argmax(add)] # br
        rearrangedPts[1] = pts[np.argmin(diff)] # tr
        rearrangedPts[2] = pts[np.argmax(diff)] # bl
        return rearrangedPts


    @staticmethod
    def getWarp(image, biggestContour, wCrop, hCrop): # image and biggest contour
        # crop
        #widthImg, heightImg = 640, 1280

        biggest = ImageProcessing.rearrangePts(biggestContour)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [wCrop, 0], [0, hCrop], [wCrop, hCrop]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(image, matrix, (wCrop, hCrop))

        imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
        imgWarped = cv2.resize(imgCropped, (wCrop, hCrop))

        return imgWarped





    @staticmethod
    def image_smoothening(image):
        BINARY_THREHOLD = 180
        ret1, th1 = cv2.threshold(image, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
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