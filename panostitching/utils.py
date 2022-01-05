import numpy as np


def shift_image_yaxis(image, dy):
    ''' Shifts an image on the y axis. All images should be cropped by the max diff (to avoid zero padded area)'''
    canvas = np.zeros_like(image)
    height = image.shape[0]
    if dy == 0: 
        return image
    elif dy > 0:
        canvas[dy:] = image[:-dy]
    elif dy < 0: 
        canvas[:dy] = image[-dy:]
    return canvas