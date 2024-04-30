import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam
from CRAB.classfication.modles.resnet_formal import ResnetBuilder

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import argparse

parser = argparse.ArgumentParser(description='Train model your dataset')
parser.add_argument('--link_place', default=0, type=int)
parser.add_argument('--re_co', default=0.01, help='regularization coefficient', type=int)
parser.add_argument('--backbone', default='resnet50', type=str)

args = parser.parse_args()
link_place = args.link_place
re_co = args.re_co
place_list = ['link_to_high_level_feature_0', 'link_to_low_level_feature_1', 'link_to_SA_2', 'link_to_ADD_3', 'no_link_4']
place_name = place_list[link_place]
path = 'dataset/Magnetic-tile-defect-datasets-master'

img_size=200
X_train = np.load('dataset/AMT_trainval_x_%s.npy' % img_size, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
y_train = np.load('dataset/AMT_trainval_y_%s.npy' % img_size, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')

X_test = np.load('dataset/AMT_test_x_%s.npy' % img_size, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
y_test = np.load('dataset/AMT_test_y_%s.npy' % img_size, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
print("successfully load x%s.npy" % img_size)

np.random.seed(7)
X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)
X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)

# # one hot encode outputs
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

X_train = X_train.reshape((len(X_train), img_size, img_size, 3))
X_test = X_test.reshape((len(X_test), img_size, img_size, 3))

if args.backbone == 'resnet50':
    model = ResnetBuilder.build_resnet_50(input_shape=(img_size, img_size, 3), num_outputs=6, link_place=link_place)
elif args.backbone == 'resnet18':
    model = ResnetBuilder.build_resnet_18(input_shape=(img_size, img_size, 3), num_outputs=6, link_place=link_place)
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
lr_reducer = ReduceLROnPlateau()

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
data_augmentation = False
from keras.preprocessing.image import ImageDataGenerator
from modles.file_restore import xls_restore
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train, Y_train,
                        batch_size=32,
                        nb_epoch=200,
                        validation_data=(X_test, Y_test),
                        shuffle=True,
                        verbose=2,
                        callbacks=[lr_reducer])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
