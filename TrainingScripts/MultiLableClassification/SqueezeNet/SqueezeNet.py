
from __future__ import print_function

import os, sys, math
import numpy as np
import pydot
import graphviz
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPool2D, GlobalAvgPool2D, Activation, Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import sys
import os, argparse

# for logging the output spitted out in the terminal
class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def fire_block(x, squeeze_fil, expand_fil):
    squeeze = Conv2D(filters=squeeze_fil, kernel_size=1, activation='relu')(x)
    expand_1 = Conv2D(filters=expand_fil, kernel_size=1, activation='relu')(squeeze)
    expand_3 = Conv2D(filters=expand_fil, kernel_size=3, activation='relu', padding='same')(squeeze)
    output = Concatenate()([expand_1, expand_3])
    return output


def main():
    # parameterizing our script. You can also run the default values to replicate our result. 
    parser = argparse.ArgumentParser(description='SqueezeNet model...')
    parser.add_argument('--train_test_split', default='10', type=str, help='The parameter splits the full training data into train and test.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=20, type=int, help="Epochs for training.")
    parser.add_argument('--num_classes', default=5, type=int, help='Number of classes for classification.')
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--data_augmentation', default=False, type=bool, help='Apply real time augmentation or not.')
    parser.add_argument('--input_shape', default='224x224x3', type=str, help="Input shape.")
    parser.add_argument('--momentum', default=0.99, type=float, help='Momentum for batch normalization.')
    parser.add_argument('--activation', default='relu', type=str, help="Activation function for CNN.")
    parser.add_argument('--path_to_data', default='./Data/Pleural Effusion_224x224_both_augmented.p', type=str, help="Pickle file location.")

    arguments = parser.parse_args()
    split = arguments.train_test_split
    epochs = arguments.epochs
    batch_size = arguments.batch_size
    num_classes = arguments.num_classes
    lr = arguments.lr
    data_augmentation = arguments.data_augmentation
    input_shape = arguments.input_shape
    input_shape = [int(i) for i in input_shape.split("x")]
    momentum = arguments.momentum
    activation = arguments.activation
    path_to_data = arguments.path_to_data

    file_name = f'SqueezeNet_sp_{split}_'\
                          f'bs_{batch_size}_'\
                          f'ep_{epochs}_'\
                          f'cls_{num_classes}_'\
                          f'lr_{lr}_'\
                          f'aug_{data_augmentation}_'\
                          f'inp_{input_shape}_'\
                          f'mom_{momentum}_'\
                          f'act_{activation}_'\
                          f'cls_{num_classes}'

    sys.stdout = Logger(f'./Logs/MultiLabelClassification/SqueezeNet/{file_name}.log')

    aug_data = pickle.load(open(path_to_data,"rb"))

    X = [pair[0].astype('float32')/255.0 for pair in aug_data]
    y = [pair[1] for pair in aug_data]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=float(split)/100, random_state=42)
    # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=splits[1], random_state=42)

    # class_weights = {0: 1-np.count_nonzero(y_train==0)/len(y_train),
    #                 1: 1-np.count_nonzero(y_train==1)/len(y_train)} 

    # y_train = utils.to_categorical(y_train, num_classes)
    # y_valid = utils.to_categorical(y_valid, num_classes)

    x = Input(input_shape)
    y = Conv2D(kernel_size=7, filters=96, strides=2, padding='same', activation=activation)(x)
    y = MaxPool2D(pool_size=3, strides=2, padding='same')(y)

    y = fire_block(y, squeeze_fil=16, expand_fil=64)
    y = fire_block(y, squeeze_fil=16, expand_fil=64)
    y = fire_block(y, squeeze_fil=32, expand_fil=128)
    y = MaxPool2D(pool_size=3, strides=2, padding='same')(y)

    y = fire_block(y, squeeze_fil=32, expand_fil=128)
    y = fire_block(y, squeeze_fil=48, expand_fil=192)
    y = fire_block(y, squeeze_fil=48, expand_fil=192)
    y = fire_block(y, squeeze_fil=64, expand_fil=256)
    y = MaxPool2D(pool_size=3, strides=2, padding='same')(y)

    y = fire_block(y, squeeze_fil=64, expand_fil=256)

    y = Conv2D(filters=num_classes, kernel_size=1, activation=activation)(y)
    y = GlobalAvgPool2D()(y)

    y = Activation("softmax")(y)

    model = Model(x, y)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()


    # The best model is selected based on the loss value on the validation set
    # filepath = "./TrainedModels/BinaryClassification/CapsNet/weights_improvement_" + file_name + "_{epoch:02d}.h5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #callbacks_list = [checkpoint]

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(
            [X_train], [y_train],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[[X_valid], [y_valid]],
            shuffle=True)
           # ,callbacks=callbacks_list)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=(0.9, 1.1),
            zoom_range=(0.85, 1.15),
            fill_mode='constant',
            cval=0.,
        )

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_valid, y_valid), shuffle=True)

    model.save(f"./TrainedModels/MultiLabelClassification/SqueezeNet/{file_name}.h5")

if __name__ == "__main__":
    main()
