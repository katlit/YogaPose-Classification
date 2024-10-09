from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from GIT
from io import BytesIO
import pandas as pd
import io
import requests



import os

from keras import layers
from keras import backend
from keras import models
from keras import utils as keras_utils
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape


import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

import keras
from keras.layers import Dense,Dropout,Conv2D,Input,MaxPool2D,Flatten,Activation, GlobalAveragePooling2D, BatchNormalization, MaxPooling2D, Conv2D, Concatenate
from keras.models import Model
keras.backend.set_image_data_format('channels_last')

#import model from keras if using single level classification

#from keras.applications.resnet import ResNet50, ResNet101
#from keras_applications.resnet_v2 import ResNet50V2, ResNet101V2
#from keras_applications.mobilenet import MobileNet
#from keras_applications.mobilenet_v2 import MobileNetV2
#from keras_applications.resnext import ResNeXt50, ResNeXt101
#from keras_applications.densenet import DenseNet121, DenseNet169, DenseNet201

# import modified densenet hirarchical model if using hirarchical classification

#from keras_densenet_modified import DenseNet201_hir

import time
import sys
import numpy as np 
np.set_printoptions(threshold=sys.maxsize)

import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

import numpy as np
from numpy import argmax
import os
import random
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import torch

import keras
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,CSVLogger
import keras.backend as K

from sklearn.metrics import classification_report

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# -*- coding:utf-8 -*-
#from models import model_one_class, dense201_hirar, dense201_hirar_6same20, dense201_hirar_new


import warnings
warnings.filterwarnings("ignore")


import os
import urllib.request
from pathlib import Path

##################################3

path = './data/Yoga5/Yoga/kaggle_yogaposes/DATASET/ALL/'

train_df = pd.read_csv(path + 'Train_df.csv')
test_df = pd.read_csv(path + 'Test_df.csv')

####################################################################


BASE_WEIGTHS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/densenet/')

DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')

def dense_block(x, blocks, name):

    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    
    x = transition_block(x, 0.5, name='pool3')
    x1=x
    x = dense_block(x, blocks[2], name='conv4')
    
    x = transition_block(x, 0.5, name='pool4')
    x2=x
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor) #tf.keras.utils.get_source_inputs
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, [x1,x2,x], name='densenet201_hir')
    
    # Load weights.
    if weights == 'imagenet':

        weights_path = keras_utils.get_file(
            'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET201_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')

        model.load_weights(weights_path)

    return model

def DenseNet201_hir(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)


def preprocess_input(x, data_format=None, **kwargs):

    return imagenet_utils.preprocess_input(x, data_format,
                                           mode='torch', **kwargs)

setattr(DenseNet201_hir, '__doc__', DenseNet.__doc__)


def dense_block(x, blocks, name):
    
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def conv_block(x, growth_rate, name):
   
    bn_axis = 3 #if backend.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def model_one_class(
        input_shape = (224,224,3),
        class_6=6,
        class_20=20,
        class_82=82):
    # for results of sota papers
    inputs = Input(input_shape)
    base_model= ResNet50(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= ResNet101(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= ResNet50V2(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= ResNet101V2(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= DenseNet121(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= DenseNet169(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= DenseNet201(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= MobileNet(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= MobileNetV2(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= ResNeXt50( input_tensor = inputs, include_top = False, weights = None,backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= DenseNet121(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    

    x=  base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(class_82, activation='softmax')(x)

    model = Model(inputs=inputs, outputs= [x])

    for layer in base_model.layers:
        layer.trainable = True
    
    return model


def dense201_hirar(
        input_shape = (224,224,3),
        class_6=6,
        class_20=20,
        class_82=82):
    
    # for variant 1 in the paper

    inputs = Input(input_shape)
    base_model= DenseNet201_hir(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
   
    [x1,x2,x] = base_model.output

    x1 = BatchNormalization( epsilon=1.001e-5, name = 'bn_class6_last')(x1)
    x1 = Activation('relu', name='relu_class6_last')(x1)                                                                                                                                                                                                                                                                    
    x1 = GlobalAveragePooling2D(name='GAvgPool_class6_last')(x1)
    x2 = BatchNormalization( epsilon=1.001e-5, name = 'bn_class20_last')(x2)
    x2 = Activation('relu', name='relu_class20_last')(x2)
    x2 = GlobalAveragePooling2D(name='GAvgPool_class20_last')(x2)
    x = GlobalAveragePooling2D()(x)

    x1 = Dense(class_6, activation= 'softmax')(x1)
    x2 = Dense(class_20, activation= 'softmax')(x2)
    x = Dense(class_82, activation='softmax')(x)

    model = Model(inputs, [x1,x2,x])

    for layer in base_model.layers:
        layer.trainable = True
    
    return model

def dense201_hirar_6same20(
        input_shape = (224,224,3),
        class_6=6,
        class_20=20,
        class_82=82):
    
    # for variant 2 in the paper
    inputs = Input(input_shape)
    base_model= DenseNet201_hir(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)

    [null,x2,x] = base_model.output

    x1 = BatchNormalization(epsilon=1.001e-5, name = 'bn_class6_last')(x2)
    x1 = Activation('relu', name='relu_class6_last')(x1)
    x1 = GlobalAveragePooling2D(name='GAvgPool_class6_last')(x1)
    x2 = BatchNormalization(epsilon=1.001e-5, name = 'bn_class20_last')(x2)
    x2 = Activation('relu', name='relu_class20_last')(x2)
    x2 = GlobalAveragePooling2D(name='GAvgPool_class20_last')(x2)
    x = GlobalAveragePooling2D()(x)

    x1 = Dense(class_6, activation= 'softmax')(x1)

    x2 = Dense(class_20, activation= 'softmax')(x2)

    x = Dense(class_82, activation='softmax')(x)

    model = Model(inputs, [x1,x2,x])

    for layer in base_model.layers:
        layer.trainable = True
    
    return model


def dense201_hirar_new(
        input_shape = (224,224,3),
        class_6=6,
        class_20=20,
        class_82=82):

    # for variant 3 in the paper

    inputs = Input(input_shape)
    base_model= DenseNet201_hir(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    
    [x1,x2,x] = base_model.output

    x1 = dense_block(x1, 32, name='denseblockClass6')


    x1 = BatchNormalization( epsilon=1.001e-5, name = 'bn_class6_last')(x1)
    x1 = Activation('relu', name='relu_class6_last')(x1)
    x1 = GlobalAveragePooling2D(name='GAvgPool_class6_last')(x1)
    x2 = BatchNormalization( epsilon=1.001e-5, name = 'bn_class20_last')(x2)
    x2 = Activation('relu', name='relu_class20_last')(x2)
    x2 = GlobalAveragePooling2D(name='GAvgPool_class20_last')(x2)
    x = GlobalAveragePooling2D()(x)

    x1 = Dense(class_6, activation= 'softmax')(x1)

    x2 = Dense(class_20, activation= 'softmax')(x2)

    x = Dense(class_82, activation='softmax')(x)

    model = Model(inputs, [x1,x2,x])

    for layer in base_model.layers:
        layer.trainable = True
    
    return model

###################################################################3



def preprocess(inputs):
    inputs /=255
    return inputs


def process_data(df,img_path,train=True):
    num = df.shape[0]
    
    data = np.zeros((num,224,224,3),dtype='float32')
    x1_labels = np.zeros(num,dtype='int')
    x2_labels = np.zeros(num,dtype='int')
    x3_labels = np.zeros(num,dtype='int')

    for i in range(num):
        path = df['YogaPoses'].iloc[i] + '/' + df['ImageNumbers'].iloc[i]
        
        x1_label = df['label of class_6'].iloc[i]
        x2_label = df['label of class_20'].iloc[i]
        x3_label = df['label of class_82'].iloc[i]

        imgs = img_path + path
        #print(imgs)
        if train:
                #response = requests.get(imgs)
                #image = Image.open(BytesIO(response.content)).convert("RGB")
                #print(image)
                #print(imgs)
                image = Image.open(imgs).convert("RGB")
                image = image.resize((224,224), Image.ANTIALIAS)
                data[i][:][:][:] = image
                x1_labels[i] = x1_label
                x2_labels[i] = x2_label
                x3_labels[i] = x3_label
            
        else:
            #response = requests.get(imgs)
            #image = Image.open(BytesIO(response.content)).convert("RGB")
            image = Image.open(imgs).convert("RGB")
            image = image.resize((224,224), Image.ANTIALIAS)
            data[i][:][:][:] = image
            x1_labels[i] = x1_label
            x2_labels[i] = x2_label
            x3_labels[i] = x3_label
    
    return data, x1_labels, x2_labels, x3_labels


def generator_train_batch(df,batch_size,num_classes,img_path):
    
    class_6 = num_classes[0]
    class_20 = num_classes[1]
    class_82 = num_classes[2]
    while True:
        
            df_shuffled = df.sample(frac=1, random_state=4)
            num = df_shuffled.shape[0]
            for i in range(0, num, batch_size):
                x_train, x1_labels, x2_labels, x3_labels = process_data(df_shuffled[i:(i+batch_size)],img_path,train=True)
                x = preprocess(x_train)
                #print(x)
                y1 = np_utils.to_categorical(np.array(x1_labels), class_6)
                y2 = np_utils.to_categorical(np.array(x2_labels), class_20)
                y3 = np_utils.to_categorical(np.array(x3_labels), class_82)
                y = [y1,y2,y3]
                yield x, y

def generator_val_batch(df,batch_size,num_classes,img_path):
    class_6 = num_classes[0]
    class_20 = num_classes[1]
    class_82 = num_classes[2]
    
    while True:
            num = df.shape[0]
            for i in range(0,num,batch_size):
                y_test,y1_labels, y2_labels, y3_labels = process_data(df[i:(i+batch_size)],img_path,train=False)
                x = preprocess(y_test)
                y1 = np_utils.to_categorical(np.array(y1_labels), class_6)
                y2 = np_utils.to_categorical(np.array(y2_labels), class_20)
                y3 = np_utils.to_categorical(np.array(y3_labels), class_82)
                test_data = x
                y = [y1,y2,y3]
                yield test_data, y


def main():
    path = './Results/Yoga5/dense201_hirar_new/'
    img_path =  './data/Yoga5/Yoga/kaggle_yogaposes/DATASET/ALL/' 

    
    train_df
    test_df
    
    num_classes = [6,20,82]
    batch_size = 32  #164 training samples, 82 test
    epochs = 20

    model = dense201_hirar_new()
    #model.load_weights('weights_betweenhirarModify_lw111_6and20ConnectSame_nopre_mix_.0003.hdf5')

    lr = 0.003 # orig= 0.003
    sgd = SGD(learning_rate=lr, momentum=0.9, nesterov=False)
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], loss_weights=[1,1,1], optimizer= sgd, metrics=['accuracy'])#, 'top_k_categorical_accuracy'])
    
    # fix random seed for reproducibility
    #seed = 7
    #tf.random.set_seed(seed)
    
    #model.summary()

    #save the Keras model or model weights at some frequency.
    # "val_loss" to monitor the model's total loss.
    #verbose: 1 displays messages when the callback takes an action.
   
    checkpointer = ModelCheckpoint(filepath=path+'Variant3.hdf5', verbose=1, save_best_only = True, monitor='val_loss')
    #Callback that streams epoch results to a CSV file.
    csv_logger= CSVLogger(path+'logV3.csv')
    #callbacks = [checkpoint]
    
    
    # get the start time
    st = time.time()
    
    history = model.fit_generator(generator_train_batch(train_df, batch_size, num_classes,img_path),
                          steps_per_epoch = train_df.shape[0] // batch_size + 1,
                          epochs=epochs,
                          callbacks=[csv_logger],
                          validation_data=generator_val_batch(test_df, batch_size,num_classes,img_path),
                          validation_steps = test_df.shape[0] // batch_size + 1,
                          verbose=1)
    
    np.save(path + 'model_weights_V3.npy',model.get_weights())
    print('FIT ENDED')


    # get the end time
    et = time.time()
    # get execution time in minutes
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')
    model = dense201_hirar_new()
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], loss_weights=[1,1,1], optimizer= sgd, metrics=['accuracy'])#, 'top_k_categorical_accuracy'])
    model.set_weights(np.load(path + 'model_weights_V3.npy', allow_pickle=True))

    batch_size = 157
    test_steps_per_epoch = test_df.shape[0] // 157
    
    score = model.evaluate_generator(generator_val_batch(test_df,batch_size,num_classes,img_path),steps=test_steps_per_epoch, verbose=1)
    np.savetxt(path + 'scoreV3.txt', score, fmt='%d')
    ###################################################################
    #generate predictions
    y_predg = model.predict_generator(generator_val_batch(test_df, batch_size, num_classes, img_path), steps=test_steps_per_epoch, verbose =1)
    np.savetxt(path + 'probabilities6_V1.txt', y_predg[0], fmt = '%d')
    np.savetxt(path + 'probabilities20_V1.txt', y_predg[1], fmt = '%d')
    np.savetxt(path + 'probabilities82_V1.txt', y_predg[2], fmt = '%d') 
    predictions6 = np.argmax(y_predg[0],axis=1)
    np.savetxt(path + 'predictions6.txt', predictions6, fmt='%d')
    predictions20 = np.argmax(y_predg[1],axis=1)
    np.savetxt(path + 'predictions20.txt', predictions20, fmt='%d')
    predictions82 = np.argmax(y_predg[2],axis=1)
    np.savetxt(path + 'predictions82.txt', predictions82, fmt='%d')
    #get true labels
    y_testlabels6 = np_utils.to_categorical(np.array(test_df['label of class_6']), 6)
    y_testlabels20 = np_utils.to_categorical(np.array(test_df['label of class_20']), 20)
    y_testlabels82 = np_utils.to_categorical(np.array(test_df['label of class_82']), 82)
    
    # confusion matrix
    cm = tf.math.confusion_matrix(np.argmax(y_testlabels6,axis=1), predictions6, 6)
    torch.save(cm, path + 'cm6.pt')
    cm = tf.math.confusion_matrix(np.argmax(y_testlabels20,axis=1), predictions20, 20)
    torch.save(cm, path + 'cm20.pt')
    cm = tf.math.confusion_matrix(np.argmax(y_testlabels82,axis=1), predictions82, 82)
    torch.save(cm, path + 'cm82.pt')
    
    #classification reports
    target_names = ['standing', 'sitting', 'inverted', 'reclining', 'wheel', 'balancing']
    report = classification_report(y_testlabels6.argmax(axis=1), predictions6, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report)
    df.to_csv(path + 'classification_report6.csv')

    target_names = ['staright', 'forward bend', 'side bend', 'others', 'fronleg', 'behindleg', 'split', 'ffw bend', 'twist', 'front', 
                    'side', 'legs straight up', 'legs bend', 'up-facing', 'down-facing', 'side-facing', 'plank balance', 'up-facing', 
                    'down-facing', 'others']
    report = classification_report(y_testlabels20.argmax(axis=1), predictions20, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report)
    df.to_csv(path + 'classification_report20.csv')
    
    
    target_names = ['1.Eagle', '2.Tree', '3.Chair','4.Standing', '5.Wide legged forward bend',
                    '6.Dolphin', '7.Downward dog', '8.Intese side stretch', '9.Half moon', '10.Extended triangle',
                    '11.Extended side angle', '12.Gate', '13.Warrior I.' , '14.ReverseWarrior', '15.Low lunges',
                    '16.Warrior II', '17.Warrior III', '18.Lord of dance', '19.Standing big-toe hold', '20.Standing split',
                    '21.Easy sitting', '22.Cobbler', '23.Garland', '24.Staff', '25.Noose', '26.Cow face', '27.Hero and thunderbolt',                
                    '28.Bhardwajas twist' , '29.Half lord of the fishes',  '30.Split', '31.Wide angle seated forward bend',
                    '32.Head to knee', '33.Revolvedhead-toknee', '34.Seated forward bend', '35.Tortoise pose',
                    '36.Shooting bow', '37.Heron', '38.King pigeon', '39.Cranecrow', '40.Shoulder pressing',
                    '41.Cockerel', '42.Scale', '43.Firefly', '44.Side crane-crow', '45.Eight angle',
                    '46.Sage Koundaniya', '47.Handstand', '48.Headstand', '49.Shoulderstand',
                    '50.Feather peacock', '51.Legs up to wall' , '52. Plow', '53.Scorpion', '54.Corpse', '55.Fish', 
                    '56.Happy baby',
                    '57.Reclining hand-to-big-toe', '58.Wind relieving', '59.Reclining cobbler', '60.Reclining hero',
                    '61.Yogic sleep',  '62.Cobra', '63.Frog', '64.Locust', '65.Child', '66.Extended puppy', '67.Side reclining leg lift',
                    '68.Side plank', '69.Dolphin plank', '70.Low plank (four limbedstaff' , '71.Plank', '72.Peacockb ', '73.Upward bow', 
                    '74.Upward facing two- foot staff', '75.Upward plank', '76.Pigeon', '77.Bridge', '78.Wild things', '79.Camel',   
                    '80.Cat-cow', '81. Boat', '82. Bow']
    report = classification_report(y_testlabels82.argmax(axis=1), predictions82, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report)
    df.to_csv(path + 'classification_report82.csv')
    
    
    # plots
    resDict = history.history
    
    def search(values, searchFor):
        keynames = []
        for k in values:
            if searchFor in k:
                    keynames.append(k)
        return keynames

    accKeys = search(resDict, 'acc')
    lossKeys = search(resDict, 'loss')
    f1Keys = search(resDict, 'f1')

    plt.rcParams["figure.figsize"] = (3,3)

    for i in range(int(len(accKeys)/2)):
        plt.plot(resDict[str(accKeys[i])])
        plt.plot(resDict[str(accKeys[(i+int(len(accKeys)/2))])])
        name=str(accKeys[i]) + ' and ' + str(accKeys[(i+int(len(accKeys)/2))])
        plt.title(name)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(path + "AccPlotV3{y}.png".format(y=i), bbox_inches='tight')
        plt.close()


    for i in range(int(len(lossKeys)/2)):
        plt.plot(resDict[str(lossKeys[i])])
        plt.plot(resDict[str(lossKeys[(i+int(len(lossKeys)/2))])])
        name = str(lossKeys[i]) + ' and ' + str(lossKeys[(i+int(len(lossKeys)/2))])
        plt.title(name)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(path + "LossPlotV3{y}.png".format(y=i), bbox_inches='tight')
        plt.close()

    
    
if __name__ == '__main__':
    main()





