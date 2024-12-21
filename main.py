from matplotlib import image
from scene import *

background_image = image.imread("background.jpg")
scene = Scene(background_image)
scene.render_animation(1, 50, 0)
