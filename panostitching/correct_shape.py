''' Script for shape correction of deformed panoramas. It corrects only y-axis, and resizes panoramas into rectangles.
Part of stitching module. 

'''

from skimage.transform import resize
from general_utils import smooth
import numpy as np
from scipy.interpolate import interp1d

def find_topNbot(img, Step_interp = 6):
    ''' Finds the top and bottom of marked images. Top is marked with a red line [255,0,0], and bot with a [0,255,0].
    
    '''
    
    x = np.arange(img.shape[1])
    diff = img[:,:, 0].astype('float') - img[:,:, 1].astype('float') # red - green will have max at top, and min at bot
    top,bot = [], []
    for i in x:
        line = diff[:,i]
        top.append(int(np.argmax(line)))
        bot.append(int(np.argmax(-line)))
    top, bot = np.array(top), np.array(bot)
    if Step_interp != 0: # Takes Step_interp points in top and bot, and interpolates between them. After this, there is
        # no discontinuity in top and bot
        
        N = int(img.shape[1]) // Step_interp
        x = np.arange(len(top))
        x_interp = np.append(x[::N], x[-1])
        top4interp, bot4interp = np.append(top[::N], top[-1]), np.append(bot[::N], bot[-1])
        ftop, fbot  = interp1d(x_interp,top4interp ), interp1d(x_interp, bot4interp )
        top_interp, bot_interp = ftop(x), fbot(x)
        return top_interp, bot_interp
    return top, bot 

def correct_shape(img, new_topbottom = [95, 1676], Step_interp = 6):
    ''' Resizes images line by line, matching top and bottom of each line (found by find_topNbot) to new_top and new_bot.
    
    '''
    
    new_bot, new_top = int(new_topbottom[1]), int(new_topbottom[0]) 
    N_new = new_bot - new_top 
    
    canvas = np.zeros_like(img)
    top, bot = find_topNbot(img, Step_interp = Step_interp) # Finds top and bot
    top, bot = np.round(top).astype('int'), np.round(bot).astype('int')
    x = np.arange(img.shape[1])
    
    for i in range(len(x)): # takes every line, between top and bot, resizes it to new_top new_bot, and sticks it into a canvas
        y_bot, y_top = bot[i], top[i]
        line = img[y_top:y_bot,i]
        new_line = resize(line, (N_new, 3))
        new_line = new_line * 255
        canvas[new_top:new_bot,i,:] = new_line.astype('uint8')
        canvas[new_top,i,:] = np.ones_like(canvas[(y_top-1),i,:]) * np.array([255,0,0])
        canvas[new_bot,i,:] = np.ones_like(canvas[(y_top-1),i,:]) * np.array([0,255,0])
    return canvas