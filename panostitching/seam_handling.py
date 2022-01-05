# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 15:04:58 2021

@author: louis.combe


Script for smart seaming of two images. DISCLAIMER : This script is built to work with open CV stitcher and requires the rightmost image to be already warped/translated + put into a canvas


"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
seam_smoothing_len = 10 
valley_to_absdiff_ratio = 0.75


def find_overlap(imageA, warped_imageB):
    idx_left = int(np.argwhere(warped_imageB[1000,:,0] > 0)[0,0])
    idx_right = imageA.shape[1]
    
    overlapA = imageA[:,idx_left:idx_right,:]
    overlapB = warped_imageB[:, idx_left:idx_right,:]

    
    return np.concatenate(([overlapA], [overlapB]))


def energy_map(overlap):
    """ makes the energy map for seam carving. Use a mix of difference
    and geometrical values. """
    
    # absolute difference -> try to cut where it's similar
    energy_absdiff = np.abs((overlap[0]-overlap[1]))

    
    # valley profile -> try to cut in the middle
    lx = float(overlap.shape[2])
    energy_valley = np.zeros(overlap[0].shape)
    for l in range(overlap.shape[1]):
        energy_valley_mono = ( (np.arange(0,lx)-lx/2)*2/lx )**2
        energy_valley[l,:] = energy_valley_mono[:,None]
    coef_valley = energy_absdiff.max() * valley_to_absdiff_ratio # to keep both comparable
    
    return enemap_3D_to_2D(energy_absdiff + coef_valley * energy_valley, 'lum')

def enemap_3D_to_2D(enemap, method = 'mean'):
    """ convert RBG enemap to greyscale. can use luminance, mean(default) or quadratic mean method """
    lx, ly = enemap.shape[:2]
    out = np.zeros((lx, ly))
    if method == 'lum':
        out = enemap[:,:,0] * 0.2126 + enemap[:,:,1] * 0.7152 + enemap[:,:,2] * 0.0722
    elif method == 'quadmean':
        out = np.mean(enemap ** 2, -1) ** 0.5
    else:
        out = enemap.mean(-1)
    return out

def find_cutting_path(overlap):
    """ find path along which to cut the overlaping region, using seam carving """
    ene_map = energy_map(overlap)
    
    return find_vertical_seam(ene_map.T)


def find_vertical_seam ( im_in ):
	"""	Takes a grayscale img and returns the lowest energy vertical seam as a list of pixels (2-tuples).
	This implements the dynamic programming seam-find algorithm. For an m*n picture, this algorithm
	takes O(m*n) time """

	im = im_in.transpose()
	u = find_horizontal_seam(im)
	for i in range(len(u)):
		temp = list(u[i])
		temp.reverse()
		u[i] = tuple(temp)
	return u


def find_horizontal_seam(im):
    """ Takes a grayscale img and returns the lowest energy horizontal seam as a list of pixels (2-tuples).
	This implements the dynamic programming seam-find algorithm. For an m*n picture, this algorithm
	takes O(m*n) time
	@im: a grayscale image
   SOURCE:https://github.com/sameeptandon/python-seam-carving/blob/master/CAIS.py
   author: Sameep Tandon """
    im_width, im_height = im.shape
    cost = np.zeros( im.shape )
    for y in range(im_height):
        cost[0,y] = im[0, y]
    for x in range(1, im_width):
        for y in range(im_height):
            if y == 0:
                min_val = min( cost[x-1,y], cost[x-1,y+1] )
            elif y < im_height - 2:
                min_val = min( cost[x-1,y], cost[x-1,y+1] )
                min_val = min( min_val, cost[x-1,y-1] )
            else:
                min_val = min( cost[x-1,y], cost[x-1,y-1] )
            cost[x,y] = im[x,y] + min_val
    min_val = np.inf
    path = [ ]
    for y in range(im_height):
        if cost[im_width-1,y] < min_val:
            min_val = cost[im_width-1,y]
            min_ptr = y

    pos = (im_width-1,min_ptr)
    path.append(pos)

    while pos[0] != 0:
        val = cost[pos] - im[pos]
        x, y = pos
        if y == 0:
            if val == cost[x-1,y+1]:
                pos = (x-1,y+1)
            else:
                pos = (x-1,y)
        elif y < im_height - 2:
            if val == cost[x-1,y+1]:
                pos = (x-1,y+1)
            elif val == cost[x-1,y]:
                pos = (x-1,y)
            else:
                pos = (x-1,y-1)
        else:
            if val == cost[x-1,y]:
                pos = (x-1,y)
            else:
                pos = (x-1,y-1)

        path.append(pos)

    return np.array(path)

def seam_fusion(i1, i2, check_border = False):
    """ Main function for smart seam fusion. Finds the best path for seaming, then fuses the images with a gaussian blur. """
    
    idx_left = int(np.argwhere(i2[1000,:,0] > 0)[0,0]) 
    idx_right = i1.shape[1]
    X1,Y1 = i1.shape[:2] ; X2,Y2 = i2.shape[:2]
    
    overlap = find_overlap(i1, i2)
    
    if overlap.shape[1] == 1: # case of 1pixel overlap
        seam = []
        for i in range(overlap.shape[0]):
            seam+=[[0,i]]
    else:
        seam = find_cutting_path(overlap)
        
    
    
    nchan = i1.shape[-1]
    output =np.zeros_like(i2, dtype='float')
    height_output, width_output = output.shape[0], output.shape[1]
    height_overlap, width_overlap = overlap[0].shape[0], overlap[0].shape[1]
    
    
    for s in seam:
        l = s[0]
        mod_left = np.zeros(width_overlap)[:,None]
        mod_left[:l] = 1
        SSL = seam_smoothing_len 
        mod_left = gaussian_filter(mod_left, SSL)
        mod_right = 1 - mod_left
        output[s[1], (idx_left):(idx_left + width_overlap)] += mod_left * overlap[0][s[1], :].astype('float')
        output[s[1], (idx_left):(idx_left + width_overlap)] += mod_right * overlap[1][s[1], :].astype('float')
        
    output[:,:idx_left] = i1[:,:idx_left]
    output[:,idx_right:] = i2[:,idx_right:]
    if check_border:
        for s in seam:
            l = s[0]
            output[s[1], l] = [255,0,0]
    return output.astype('uint8')