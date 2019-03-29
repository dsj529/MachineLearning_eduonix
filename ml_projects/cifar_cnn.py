'''
Created on Mar 28, 2019

@author: dsj529

implementation of the all-CNN topology defined in the paper at https://arxiv.org/pdf/1412.6806.pdf
pre-trained weights obtained from https://github.com/PAN001/All-CNN
the cifar10 dataset is maintained and described at https://www.cs.toronto.edu/~kriz/cifar.html
'''
from PIL import Image
from keras.datasets import cifar10
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from matplotlib import pyplot as plt
from matplotlib.pyplot import title

import numpy as np


# load the test and train datasets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#===================================================================================================
# ## inspect the data
# print('Training Images: {}'.format(X_train.shape))
# print('Testing Images: {}'.format(X_test.shape))
# ## inspect a single image matrix
# print(X_train[0].shape)
# ## show a 3x3 grid of sample images
# for i in range(9):
#     plt.subplot(330+1+i)
#     img = X_train[i]
#     plt.imshow(img)
# plt.show()
#===================================================================================================

## start preprocessing the data
X_train = X_train.astype('float32') / 225.
X_test = X_test.astype('float32') / 225.
## look at class labels set
# print(y_train.shape)
# print(y_train[0])

## hot-encode the output classes
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)
# print(y_train[0])


def allcnn(weights=None):
    model = Sequential()
    # start adding layers
    model.add(Conv2D(96, (3,3), padding='same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1,1), padding='same'))
    
    # cap it with a global pooling layer / softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    # load weights if provided
    if weights:
        model.load_weights(weights)
        
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# function to plot accuracy and loss
def plot_model_history(model_history):
    ''' this function found at parneetk.github.com/blog/cifar-10 '''
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
        # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
#===================================================================================================
# code to explicitly train the model is commented out, due to excessive resource demands
# on a CPU-only implementation.
#
# model = allcnn()
# model_info = model.fit(X_train, y_train,
#                        batch_size=128, epochs=200,
#                        validation_data=(X_test, y_test),
#                        verbose=1)
# 
# plot_model_history(model_info)
#===================================================================================================

## compile with pre-trained weights file
model = allcnn('../data/all_cnn_weights_0.9088_0.4994.hdf5')
print(model.summary())
scores = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: {:.3%}'.format(scores[1]))

## now to start predicting!
classes = range(10)
names = ['airplane', 'automobile',
         'bird', 'cat', 'deer', 'dog',
         'frog', 'horse', 'ship', 'truck']

class_labels = dict(zip(classes, names))

test_batch = X_test[100:109]
labels = np.argmax(y_test[100:109], axis=-1)

preds = model.predict(test_batch, verbose=1)
for pred in preds:
    print(np.sum(pred))
class_result=np.argmax(preds, axis=-1)
print(class_result)

fig, axs = plt.subplots(3,3, figsize=(15,6))
fig.subplots_adjust(hspace=1)
axs = axs.flatten()

for i, img in enumerate(test_batch):
    for k, v in class_labels.items():
        if class_result[i] == k:
            title = 'Prediction: {}\nActual: {}'.format(class_labels[k], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)
            
    axs[i].imshow(img)
plt.show()