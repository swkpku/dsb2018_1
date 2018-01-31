import os
import sys
from skimage.io import imread, imsave
from skimage.measure import regionprops
from tqdm import tqdm
import numpy as np
from skimage import img_as_uint

TRAIN_ROOT = '/home/swk/dsb2018/stage1_train/'

IMG_CHANNELS = 3

# Get train and test IDs
train_ids = next(os.walk(TRAIN_ROOT))[1]

print('Getting train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_ROOT + id_
    bbox_path = path
    bbox_file = open(path + "/bbox.txt", 'w')
    bbox = list()
        
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        
        props = regionprops(mask_)
        
        if len(props) != 1:
            print(len(props))
            print(id_)

        #IMG_HEIGHT, IMG_WIDTH = mask_.shape[0], mask_.shape[1]
        
        #ymin = IMG_HEIGHT
        #xmin = IMG_WIDTH
        #ymax = 0
        #xmax = 0

        #for x in range(IMG_HEIGHT):
        #    for y in range(IMG_WIDTH):
        #        if mask_[x,y] == 1:
        #            ymin = min(ymin, y)
        #            xmin = min(xmin, x)
        #            ymax = max(ymax, y)
        #            xmax = max(xmax, x)
                    
        bbox_file.write(str(props[0].bbox[0])+' ')
        bbox_file.write(str(props[0].bbox[1])+' ')
        bbox_file.write(str(props[0].bbox[2])+' ')
        bbox_file.write(str(props[0].bbox[3])+'\n')
    
