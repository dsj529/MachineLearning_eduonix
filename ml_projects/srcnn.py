'''
Created on Mar 28, 2019

@author: dsj529

implements the SRCNN topology for image enhancement and reconstruction.
The network is defined in a paper at https://arxiv.org/abs/1501.00092
Image files are hosted at http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
network weights downloaded from https://github.com/MarkPrecursor/SRCNN-keras
'''
import math
import os, os.path

import cv2
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim

import numpy as np


BASE_PATH = '../data/srcnn_data/'

## define metric functions
def psnr(target, ref):
    '''peak signal-to-noise ratio.  Lower is better'''
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    
    diff = ref_data - target_data
    diff = diff.flatten('C')
    
    rmse = math.sqrt(np.mean(diff ** 2.))
    
    return 20 * math.log10(255. / rmse)

def mse(target, ref):
    '''mean squared error.  Lower is better'''
    err = np.sum((target.astype('float') - ref.astype('float')) **2)
    err /= float(target.shape[0] * target.shape[1])
    return err

def compare_images(target, ref):
    '''helper function to call all quality metrics'''
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel=True))
    return scores

## prepare the images for the NN
def prepare_images(factor):
    src = os.path.join(BASE_PATH, 'originals')
    for file in os.listdir(os.path.join(src)):
        img = cv2.imread(os.path.join(src, file))
        
        # determine old and new image dimensions
        h, w, _ = img.shape
        new_height = h // factor
        new_width = w // factor
        
        # resize image down
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # resize image back up
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        print('Saving: {}'.format(file))
        cv2.imwrite(os.path.join(BASE_PATH, 'degraded', file), img)
        
prepare_images(2)

## compare originals to degraded low-res images
for file in os.listdir(os.path.join(BASE_PATH, 'originals')):
    target = cv2.imread(os.path.join(BASE_PATH, 'degraded', file))
    ref = cv2.imread(os.path.join(BASE_PATH, 'originals', file))
    
    scores = compare_images(target, ref)
    
    print('{}\n\tPSNR: {}\n\tMSE: {}\n\tSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))
    
## build the SRCNN model
def model():
    SRCNN = Sequential()
    
    SRCNN.add(Conv2D(filters=128, kernel_size=(9,9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    
    adam = Adam(lr=0.0003)
    
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN
    
## define some helper functions
def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img

def shave(image, border):
    img = image[border:-border, border:-border]
    return img

## define the main prediction function
def predict(fname):
    srcnn = model()
    srcnn.load_weights(os.path.join(BASE_PATH, '3051crop_weight_200.h5'))
    
    # load the degraded and reference images
    degraded = cv2.imread(os.path.join(BASE_PATH, 'degraded', fname))
    ref = cv2.imread(os.path.join(BASE_PATH, 'originals', fname))
    
    # preprocess the image
    ref = modcrop(ref, 3)
    degraded = modcrop(degraded, 3)
    
    # convert to YCrCb color space -- SRCNN model trained on Y channel
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float)/ 255
    
    pred = srcnn.predict(Y, batch_size=1)
    
    # post-process prediction
    pred *= 255
    pred[pred[:] > 255] = 255
    pred[pred[:] < 0] = 0
    pred = pred.astype(np.uint8)
    
    # return the Y channel to image and convert back to BGR
    temp = shave(temp, 6)
    temp[:, :, 0] = pred[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    
    ref = shave(ref.astype(np.uint8), 6)
    degraded = shave(degraded.astype(np.uint8), 6)
    
    # calcualte image metric scores
    scores = []
    scores.append(compare_images(degraded, ref))
    scores.append(compare_images(output, ref))
    return ref, degraded, output, scores

#===================================================================================================
# ref, degraded, output, scores = predict('flowers.bmp')
# print('Degraded Image:\n\tPSNR: {}\n\tMSE: {}\n\tSSIM: {}\n'
#       .format(scores[0][0], scores[0][1], scores[0][2]))
# print('Reconstructed Image:\n\tPSNR: {}\n\tMSE: {}\n\tSSIM: {}\n'
#       .format(scores[1][0], scores[1][1], scores[1][2]))
# 
# # display images as subplots
# fig, axs = plt.subplots(1, 3, figsize=(20, 8))
# axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
# axs[0].set_title('Original')
# axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
# axs[1].set_title('Degraded')
# axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
# axs[2].set_title('SRCNN')
# 
# # remove the x and y ticks
# for ax in axs:
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()
#===================================================================================================

for file in os.listdir(os.path.join(BASE_PATH, 'originals')):
    ref, degraded, output, scores = predict(file)
    
    # display images as subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Degraded')
    axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[2].set_title('SRCNN')
    
    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        
    fname, _ = os.path.splitext(file)
    print('Saving {}'.format(fname))
    fig.savefig(os.path.join(BASE_PATH, 'results', '{}.png'.format(fname)))