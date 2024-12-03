import numpy as np
from utilities import *

class Camera:
    def __init__(self, position, up, lookat, focal_length, cam_width, im_width, im_ratio, background_image):
        self.position = position
        self.focal_length = focal_length

        self.w = normalize(position - lookat)
        self.v = normalize(up - np.dot(up, self.w) * self.w)
        self.u = np.cross(self.v, self.w)

        self.width = cam_width
        self.height = self.width / im_ratio

        self.im_width = im_width
        self.im_height = np.round(im_width / im_ratio)

        self.pixel_u = self.u / self.im_width
        self.pixel_v = self.v / self.im_height

        self.top = self.position - self.focal_length * self.w - (self.width / 2) * self.u - (self.height / 2) * self.v

        self.background_image = background_image



