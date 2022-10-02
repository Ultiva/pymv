import cv2

__all__ = ['SuperResolution']

class SuperResolution(object):
    # https://github.com/opencv/opencv_contrib/tree/a26f71313009c93d105151094436eecd4a0990ed/modules/dnn_superres
    # https://learnopencv.com/super-resolution-in-opencv/#sec4
    def __init__(self, srModel, magnification):
        self.srModel = srModel
        self.magnification = magnification
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()

        try:
            srPath = f"./superresolution/srmodels/{self.srModel}_x{self.magnification}.pb"
            self.sr.readModel(srPath)
        except:
            print(f"Super Resolution model not found. Fallback to espcn_x2")
            self.srModel = "ESPNC"
            self.sr.readModel("./superresolution/srmodels/espcn_x2.pb")

        self.sr.setModel(f"{self.srModel.lower()}", self.magnification)

    def upsample(self, img):
        return self.sr.upsample(img)