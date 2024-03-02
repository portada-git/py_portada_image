from skimage import io as skio
from skimage.color import rgb2gray
import numpy as np
from deskew import determine_skew


class deskew_tool(object):
    """
    This class allow to fix an image document with its lines skewed.
    USE EXAMPLE:
    from py_portada_image.deskew_tools import deskew_tool

    dsk = deskew_tool(input_path="path/image", minAngle=5)
    if dsk.isSkewed():
        dsk.deskew()
        dsk.saveImage('new_image_path')
    """
    def __init__(self, input_path='', minAngle=0.1):
        self.__grayscale = None
        self.__minAngle = minAngle
        if len(input_path) > 0:
            self._image_path = input_path
            self._image = skio.imread(input_path)
        else:
            self.image = None
            self._image_path = ''

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        self._image = val

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, val):
        self._image_path = val
        self._image = skio.imread(val)
        self.__grayscale = None

    @property
    def minAngle(self):
        return self.__minAngle

    @minAngle.setter
    def minAngle(self, val):
        self.__minAngle = val

    def verifyImage(self):
        if self.image is None:
            raise Exception("Error: Image is not specified.")

    def rgb2gray(self):
        if self.__grayscale is None:
            self.verifyImage()
            if len(self.image.shape) > 2:
                self.__grayscale = rgb2gray(self.image)
            else:
                self.__grayscale = np.copy(self.image)

    def readImage(self, path):
        self.image_path = path

    def saveImage(self, image_path=''):
        if len(image_path)==0:
            image_path = self.image_path
        skio.imsave(image_path, self.image)

    def isSkewed(self):
        angle = determine_skew(self.__grayscale)
        return angle >= self.minAngle

    def deskewImage(self):
        self.verifyImage()
        if len(self.image.shape) > 2:
            self.rgb2gray()
        angle = determine_skew(self.__grayscale)
        if angle >= self.minAngle:
            rotated = rotate(self.image, angle, resize=True) * 255
            self.image = rotated.astype(np.uint8)

dsk = deskew_tool

def deskewSkimage(skimage):
    """
    This function deskew the image passed by parameter
    :param skimage: image read by skimage.io.imread
    :return: the image fixed
    """
    dsk.image=skimage
    dsk.deskewImage()
    return dsk.image

def isSkimageSkewed(skimage, minAngle=0):
    """
    This function return if the image passed by parameter is 
    more skewed then the minAngle indicated by teh second parameter
    :param minAngle: Minimum angle from which correction is required
    :param skimage: image read by skimage.io.imread
    :return: 
    """
    dsk.image=skimage
    dsk.minAngle = minAngle
    return dsk.isSkewed()

def deskewImageFile(input_path, output_path=''):
    """
    This function read the image file passed as the first parameter,
    deskew the image and save the fixed image in output_path if it is
    not empty or using the same input_path elsewhere.
    :param input_path: path where thw image is
    :param output_path: path to be used to save teh fixed image. By default
    output_path = input_path
    :return: None
    """
    dsk.image_path=input_path
    dsk.deskewImage()
    dsk.saveImage(output_path)