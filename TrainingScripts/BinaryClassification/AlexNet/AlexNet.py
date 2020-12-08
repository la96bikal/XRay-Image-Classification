import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tqdm import tqdm
import argparse
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import sys
import numpy as np

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

class AlexNet(tf.keras.Model):
    
    def __init__(self, num_classes, image_size, optimizer):
        super(AlexNet, self).__init__()
        
        self.input_size = image_size
        self.optimizer = optimizer

        #We use this list for graphing loss per batch
        self.loss_list = []
        
        #1st Convolution Layer
        self.conv1 = Conv2D(filters = 96, input_shape = image_size, kernel_size = (11,11), strides = (4,4), padding = 'valid')
        self.act1 = Activation('relu')        
        self.max1 = MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'valid')

        #2nd Convolution Layer
        self.conv2 = Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), padding = 'valid')
        self.act2 = Activation('relu')
        self.max2 = MaxPooling2D(pool_size=(3,3), strides = (2,2), padding='valid')
              
        #3rd Convlutional Layer
        self.conv3 = Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), padding = 'valid')
        self.act3 = Activation('relu')

        #4th Convolutional layer
        self.conv4 = Conv2D(filters=384, kernel_size = (3,3), strides = (1,1), padding = 'valid')
        self.act4 = Activation('relu')

        #5th Convlutional Layer
        self.conv5 = Conv2D(filters=256, kernel_size = (3,3), strides = (1,1), padding = 'valid')
        self.act5 = Activation('relu')       
        self.max5 = MaxPooling2D(pool_size=(3,3), strides = (2,2), padding='valid')

        self.flatten = Flatten()
        
        #Dense Layers
        self.dense6 = Dense(4096 + 2, input_shape = (224*224*3,))
        self.act6 = Activation('relu')
        self.do6 = Dropout(0.4)
            
        self.dense7 = Dense(4096)
        self.act7 = Activation('relu')
        self.do7 = Dropout(0.4)

        self.Dense8 = Dense(num_classes)
        self.output8 = Activation('softmax')
        
    def call(self, inputs):
        layer1 = self.conv1(inputs)
        layer1 = self.act1(layer1)
        layer1 = self.max1(layer1)
        
        layer2 = self.conv2(layer1)
        layer2 = self.act2(layer2)
        layer2 = self.max2(layer2)
        
        layer3 = self.conv3(layer2)
        layer3 = self.act3(layer3)
        
        layer4 = self.conv4(layer3)
        layer4 = self.act4(layer4)
        
        layer5 = self.conv5(layer4)
        layer5 = self.act5(layer5)
        layer5 = self.max5(layer5)
        
        flattened_layer = self.flatten(layer5)
        
        layer6 = self.dense6(flattened_layer)
        layer6 = self.act6(layer6)
        layer6 = self.do6(layer6)
        
        layer7 = self.dense7(layer6)
        layer7 = self.act7(layer7)
        layer7 = self.do7(layer7)
        
        layer8 = self.Dense8(layer7)
        layer8 = self.output8(layer8)
        
        return layer8   
    
    def accuracy(self, logits, labels):       
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
   
    def test_accuracy(self, images, labels):
        test_acc = []
        for i in range(0, len(images), 256):            
            test_batch = images[i:i + 256]        
            label_batch = labels[i:i + 256]                                                                                                                              
            test_acc.append(self.accuracy(self(test_batch), label_batch))
        return np.average(test_acc)
    
    def loss(self, logits, labels):
        return tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    
    def train(self, train_inputs, train_labels, epochs = 1, batch_size = None, validation_set = None):
        BATCH_SZ = batch_size        
        losses = []        
        train_acc = []
        
        for j in range(epochs):
            print('Epoch :: ', j + 1)
            pbar = tqdm(total = len(train_inputs))
            for i in range(0, len(train_inputs), BATCH_SZ):            
                image = train_inputs[i:i + BATCH_SZ]        
                label = train_labels[i:i + BATCH_SZ]
                # Implement backprop:

                with tf.GradientTape() as tape:
                    predictions = self(image) # this calls the call function conveniently                   
                    loss = self.loss(predictions, label)          
                losses.append(loss[0])
                
                pbar.update(BATCH_SZ)            
                
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))   

                train_acc.append(self.accuracy(self(image), label))
            
            print("Training Accuracy ::", np.average(train_acc))
            if validation_set != None:
                print("Validation Accuracy ::", self.test_accuracy(validation_set[0], validation_set[1]))
                
            self.loss_list.append(np.average(losses))
            pbar.close()   

def train_test_splitter(dataset, split):
    X,Y = map(list,zip(*dataset))

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.1)
    dataset = []
    y_train =to_categorical(y_train)
    y_test =to_categorical(y_test)

    X_train = np.array(X_train, dtype = np.float32) / 255.0
    X_test = np.array(X_test, dtype = np.float32) / 255.0     

    return X_train, X_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser(description='AlexNet Binary Model')
    parser.add_argument('--path_to_data', default='./Data/Pleural Effusion_224x224_both_augmented.p', type=str, help="Pickle file location.")
    parser.add_argument('--epochs', default=5, type=int, help="Epochs for training.")
    parser.add_argument('--train_test_split', default='0.1', type=str, help='The parameter splits the full training data into train and test.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--lr', default=0.0001, type=float, help="Learning rate")
    parser.add_argument('--input_shape', default='224x224x3', type=str, help="Input shape.")

    arguments = parser.parse_args()
    split = arguments.train_test_split
    epochs = arguments.epochs
    batch_size = arguments.batch_size
    lr = arguments.lr
    path_to_data = arguments.path_to_data
    image_size = arguments.input_shape

    file_name = f'AlexNet_sp_{split}_'\
                          f'bs_{batch_size}_'\
                          f'ep_{epochs}_'\
                          f'lr_{lr}_'\
                          f'inp_{image_size}_'\

    sys.stdout = Logger(f'./Logs/BinaryClassification/AlexNet/{file_name}.log')

    dataset = pickle.load(open(path_to_data,'rb'))
    X_train, X_test, y_train, y_test = train_test_splitter(dataset, float(split))
    model = AlexNet(2, image_size, optimizer = tf.keras.optimizers.Adam(learning_rate = lr))
    model.train(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_set = (X_test, y_test))


    # model.save_weights('AlexNet85-78.h5')

if __name__ == "__main__":
    main()
