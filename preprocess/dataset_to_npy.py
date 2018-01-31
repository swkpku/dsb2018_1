import os
import sys
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold

TRAIN_ROOT = '/home/swk/dsb2018/stage1_train/'
TEST_ROOT = '/home/swk/dsb2018/stage1_test/'

TRAIN_DATA_ROOT = '/home/swk/dsb2018/stage1_train_data/'
TEST_DATA_ROOT = '/home/swk/dsb2018/stage1_test_data/'

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Get train and test IDs
train_ids = next(os.walk(TRAIN_ROOT))[1]
test_ids = next(os.walk(TEST_ROOT))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_ROOT + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask
    
np.save(TRAIN_DATA_ROOT+'X_train_256', X_train)
np.save(TRAIN_DATA_ROOT+'Y_train_256', Y_train)
	
# split train val set
kf = KFold(n_splits=5)
split_i = 0
for train_index, val_index in kf.split(X_train):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    Y_train_train, Y_train_val = Y_train[train_index], Y_train[val_index]
    train_ids_train, train_ids_val = [train_ids[i] for i in train_index], [train_ids[i] for i in val_index]
    
    np.save(TRAIN_DATA_ROOT+'X_train_256_'+str(split_i), X_train_train)
    np.save(TRAIN_DATA_ROOT+'X_val_256_'+str(split_i), X_train_val)
    np.save(TRAIN_DATA_ROOT+'Y_train_256_'+str(split_i), Y_train_train)
    np.save(TRAIN_DATA_ROOT+'Y_val_256_'+str(split_i), Y_train_val)
    
    train_ids_train_file = open(TRAIN_DATA_ROOT+'train_ids_train_256_'+str(split_i)+'.txt', 'w')
    for id in train_ids_train:
        train_ids_train_file.write('%s\n' % id)
        
    train_ids_val_file = open(TRAIN_DATA_ROOT+'train_ids_val_256_'+str(split_i)+'.txt', 'w')
    for id in train_ids_val:
        train_ids_val_file.write('%s\n' % id)
    
    split_i = split_i+1

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_ROOT + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

np.save(TEST_DATA_ROOT+'X_test_256', X_test)

test_ids_file = open(TEST_DATA_ROOT+'test_ids_256.txt', 'w')
for id in test_ids:
    test_ids_file.write('%s\n' % id)
    
sizes_test_file = open(TEST_DATA_ROOT+'sizes_test.txt', 'w')
for sizes in sizes_test:
    for size in sizes:
        sizes_test_file.write('%d ' % size)
    sizes_test_file.write('\n')
