from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import sys
import os, argparse
K.set_image_data_format('channels_last')

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

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)


class Capsule(Layer):

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,  
        })
        return config

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def main():
    # parameterizing our script. You can also run the default values to replicate our result. 
    parser = argparse.ArgumentParser(description='CapsNet model...')
    parser.add_argument('--train_test_split', default='10', type=str, help='The parameter splits the full training data into train and test.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=20, type=int, help="Epochs for training.")
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes for classification.')
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
    momentum = arguments.momentum
    activation = arguments.activation
    path_to_data = arguments.path_to_data

    file_name = f'CapsNet_sp_{split}_'\
                          f'bs_{batch_size}_'\
                          f'ep_{epochs}_'\
                          f'cls_{num_classes}_'\
                          f'lr_{lr}_'\
                          f'aug_{data_augmentation}_'\
                          f'inp_{input_shape}_'\
                          f'mom_{momentum}_'\
                          f'act_{activation}'

    sys.stdout = Logger(f'./Logs/BinaryClassification/CapsNet/{file_name}.log')

    aug_data = pickle.load(open(path_to_data,"rb"))

    X = [pair[0].astype('float32')/255.0 for pair in aug_data]
    y = [pair[1] for pair in aug_data]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=float(split)/100, random_state=42)
    # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=splits[1], random_state=42)

    class_weights = {0: 1-np.count_nonzero(y_train==0)/len(y_train),
                    1: 1-np.count_nonzero(y_train==1)/len(y_train)} 

    y_train = utils.to_categorical(y_train, num_classes)
    y_valid = utils.to_categorical(y_valid, num_classes)

    input_image = Input(shape=(None, None, 3))
    x = Conv2D(64, (3, 3), activation=activation)(input_image)
    x=BatchNormalization(axis=-1, momentum=momentum, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
    x = Conv2D(64, (3, 3), activation=activation)(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation=activation)(x)
    x = Conv2D(128, (3, 3), activation=activation)(x)

    x = Reshape((-1, 128))(x)
    x = Capsule(32, 8, 3, True)(x)  
    x = Capsule(32, 8, 3, True)(x)   
    capsule = Capsule(num_classes, 16, 3, True)(x)
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

    model = Model(inputs=[input_image], outputs=[output])

    adam = optimizers.Adam(lr=lr) 

    model.compile(loss=margin_loss, optimizer=adam, metrics=['accuracy'])
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
            validation_data=[[X_valid], [y_valid]], class_weight=class_weights,
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

if __name__ == "__main__":
    main()
