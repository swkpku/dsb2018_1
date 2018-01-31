import os
import sys
from skimage.io import imread, imsave
from tqdm import tqdm
import numpy as np
from skimage import img_as_uint

TRAIN_ROOT = '/home/swk/dsb2018/stage1_train/'
TEST_ROOT = '/home/swk/dsb2018/stage1_test/'

IMG_CHANNELS = 3

# Get train and test IDs
train_ids = next(os.walk(TRAIN_ROOT))[1]
test_ids = next(os.walk(TEST_ROOT))[1]

print('Getting train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_ROOT + id_
    one_mask_path = path + '/one_mask/'
    
    if not os.path.exists(one_mask_path):
        os.makedirs(one_mask_path)
        
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    imsave(one_mask_path + id_ + '.png', img)
    
    IMG_HEIGHT, IMG_WIDTH = img.shape[0], img.shape[1]
    
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask = np.maximum(mask, mask_)
        
    mask = img_as_uint(mask)
    imsave(one_mask_path + 'one_mask.png', mask)
