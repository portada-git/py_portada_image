#==========================================
# Title:  Deskew tools for images
# Project: PortADa project
# Author: Josep Cañellas
# Date:   Mach 2024
#==========================================


from skimage import io as sk_io
from skimage.color import rgb2gray
import numpy as np
from deskew import determine_skew
from skimage.transform import rotate
from PIL import Image


class DeskewTool(object):
    """
    This class allow to fix an image document with its lines skewed.
    USE EXAMPLE:
    from py_portada_image.deskew_tools import DeskewTool

    dsk = DeskewTool(input_path="path/image", min_angle=5)
    if dsk.is_skewed():
        dsk.deskew()
        dsk.save_image('new_image_path')
    """

    def __init__(self, input_path='', min_angle=0.1):
        self.__grayscale = None
        self.__minAngle = min_angle
        if len(input_path) > 0:
            self._image_path = input_path
            self._image = sk_io.imread(input_path)
        else:
            self.image = None
            self._image_path = ''

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        self._image = val
        self.__grayscale = None

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, val):
        self._image_path = val
        self.image = sk_io.imread(val)

    @property
    def min_angle(self):
        return self.__minAngle

    @min_angle.setter
    def min_angle(self, val):
        self.__minAngle = val

    def __verifyImage(self):
        if self.image is None:
            raise Exception("Error: Image is not specified.")

    def add_margin(self, top, right, bottom, left):
        pil_img = Image.fromarray(self.image)
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        if pil_img.mode=='L':
            color = 255
        else:
            color = (255, 255, 255)
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        result = np.array(result)
        return result

    def rgb2gray(self):
        """
        convert the image of image attribute to grayscale and save it in __grayscale attribute
        :return: None
        """
        if self.__grayscale is None:
            self.__verifyImage()
            if len(self.image.shape) > 2:
                self.__grayscale = rgb2gray(self.image)
            else:
                self.__grayscale = np.copy(self.image)

    def read_image(self, path):
        """
        read the image from 'path' file
        :param path: the path where the image is
        :return: None
        """
        self.image_path = path

    def save_image(self, image_path=''):
        """
        Save the image from 'self.image' to 'image_path'. By default, image_path is equal to 'self.image_path'
        :param image_path: the image path where save the image
        :return: None
        """
        self.__verifyImage()
        if len(image_path) == 0:
            image_path = self.image_path
        sk_io.imsave(image_path, self.image)

    def is_skewed(self):
        """
        Calculate if image is skewed in an angle bigger than self.min_angle
        :return:
        """
        self.__verifyImage()
        self.rgb2gray()
        angle = determine_skew(self.__grayscale)
        return abs(angle) >= self.min_angle

    def deskew_image(self):
        """
        Deskew the image from self.image.
        :return:None
        """
        self.__verifyImage()
        self.rgb2gray()
        angle = determine_skew(self.__grayscale, max_angle=44, min_angle=-44)
        if abs(angle) >= self.min_angle:
            image = self.add_margin(0, 30, 0, 30)
            rotated = rotate(image, angle, resize=True) * 255
            self.image = rotated.astype(np.uint8)


dsk = DeskewTool()


def deskew_skimage(skimage):
    """
    This function deskew the image passed by parameter
    :param skimage: image read by skimage.io.imread
    :return: the image fixed
    """
    dsk.image = skimage
    dsk.deskew_image()
    return dsk.image


def is_skimage_skewed(skimage, min_angle=0):
    """
    This function return if the image passed by parameter is 
    more skewed than the min_angle indicated by the second parameter
    :param min_angle: Minimum angle from which correction is required
    :param skimage: image read by skimage.io.imread
    :return: 
    """
    dsk.image = skimage
    dsk.min_angle = min_angle
    return dsk.is_skewed()


def deskew_image_file(input_path, output_path=''):
    """
    This function read the image file passed as the first parameter,
    deskew the image and save the fixed image in output_path if it is
    not empty or using the same input_path elsewhere.
    :param input_path: path where thw image is
    :param output_path: path to be used to save teh fixed image. By default, output_path
    has the same value as input_path
    output_path = input_path
    :return: None
    """
    dsk.image_path = input_path
    dsk.deskew_image()
    dsk.save_image(output_path)
