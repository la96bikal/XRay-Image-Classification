import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tqdm import tqdm
import argparse
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from datetime import datetime

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
        self.batch_loss_list = []
        self.batch_accuracy_list = []
        self.epoch_accuracy_list = []
        self.epoch_validation_list = []
        self.confusion_m = np.zeros((2,2))
        
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

        self.concatenate = Concatenate(axis = 1)
        #Dense Layers
        self.dense6 = Dense(5000, input_shape = (224*224*3,))
        self.act6 = Activation('relu')
        self.do6 = Dropout(0.4)
            
        self.dense7 = Dense(4098)
        self.act7 = Activation('relu')
        self.do7 = Dropout(0.4)

        self.Dense8 = Dense(num_classes)
        self.output8 = Activation('softmax')
        
    def call(self, images, sex_age):         
        layer1 = self.conv1(images)
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
        concatenate_layer = self.concatenate([flattened_layer, sex_age])       
        
        layer6 = self.dense6(concatenate_layer)

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
            x_batch = images[i: i + 256]
            x_images, x_sex_age = homogenize(x_batch)                        
            label_batch = tf.Variable(labels[i:i + 256])                                                                                                                              
            test_acc.append(self.accuracy(self(x_images, x_sex_age), label_batch))
        return np.average(test_acc)
    
    def loss(self, logits, labels):
        return tf.nn.softmax_cross_entropy_with_logits(labels, logits)

    def calculate_confusion_matrix(self, logits, labels):
        y_pred = (tf.argmax(logits, 1))
        y_true = tf.argmax(labels, 1)        
        conf_matrix = confusion_matrix(y_true, y_pred)
        self.confusion_m = self.confusion_m + conf_matrix
        return None

    
    def train(self, train_inputs, train_labels, epochs = 1, batch_size = None, validation_set = None):          
        BATCH_SZ = batch_size        
        losses = []                
        train_acc = []

        for j in range(epochs):            
            print('Epoch :: ', j + 1)
            train_acc = []
            pbar = tqdm(total = len(train_inputs))
            for i in range(0, len(train_inputs), BATCH_SZ):                
                x_images, x_sex_age = homogenize(train_inputs[i: i + BATCH_SZ])                           
                label = train_labels[i:i + BATCH_SZ]
                # Implement backprop:

                with tf.GradientTape() as tape:                                         
                    predictions = self(x_images, x_sex_age) # this calls the call function conveniently                   
                    loss = self.loss(predictions, label)           
                
                pbar.update(BATCH_SZ)            
                
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))   

                batch_accuracy = self.accuracy(self(x_images, x_sex_age), label)
                train_acc.append(batch_accuracy)
                
                self.calculate_confusion_matrix(predictions, label)
                self.batch_accuracy_list.append(batch_accuracy)
                self.batch_loss_list.append(np.average(loss))

            print("Training Accuracy ::", np.average(train_acc))

            if validation_set != None:
                validation_accuracy = self.test_accuracy(validation_set[0], validation_set[1])
                print("Validation Accuracy ::", validation_accuracy)
                self.epoch_validation_list.append(validation_accuracy)

            self.epoch_accuracy_list.append(np.average(train_acc))
                    
            pbar.close()

        return np.average(train_acc)   

def train_test_splitter(dataset, split):
    X,Y = map(list,zip(*dataset))

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.1)

    del X
    del Y
    dataset = []
    y_train =to_categorical(y_train)
    y_test =to_categorical(y_test)
            
    X_train_np = []
    for i in range(len(X_train)):  
        element = X_train.pop(0)   
        row = []           
        row.append(element[0] / 255.0)
        row.append(element[1] / 1.0)
        row.append(element[2] / 1.0)
        X_train_np.append(row)

    X_test_np = []
    for i in range(len(X_test)): 
        element = X_test.pop(0)   
        row = []           
        row.append(element[0] / 255.0)
        row.append(element[1] / 1.0)
        row.append(element[2] / 1.0)
        X_test_np.append(row)        
       

    return X_train_np, X_test_np, y_train, y_test

def homogenize(X):
    images, sex, age = zip(*X)    
    sex_age = list(zip(sex, age))
    return np.array(images), np.array(sex_age)

def main():
    parser = argparse.ArgumentParser(description='AlexNet Binary Model')
    parser.add_argument('--path_to_data', default='./Data/Atelectasis_224x224_both_augmented.p', type=str, help="Pickle file location.")
    parser.add_argument('--epochs', default=5, type=int, help="Epochs for training.")
    parser.add_argument('--train_test_split', default='0.1', type=str, help='The parameter splits the full training data into train and test.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--lr', default=0.00002, type=float, help="Learning rate")
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
    X_image, X_sex, X_age, Y = zip(*dataset)
    del dataset
    X = list(zip(X_image, X_sex, X_age)) 
    X_Y = zip(X, Y)
    del X
    del Y

    X_train, X_test, y_train, y_test = train_test_splitter(X_Y, 0.1)
    del X_Y

    model = AlexNet(2, image_size, optimizer = tf.keras.optimizers.Adam(learning_rate = lr))
    model.train(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_set = (X_test, y_test))

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    graph_path = './GraphingArtifacts/'
    directory = f'AlexNet_{model.epoch_accuracy_list[-1]}_{model.epoch_validation_list[-1]}_{dt_string}_pleuralEffusion'

    dir_path = os.path.join(graph_path, directory) 
    os.mkdir(dir_path)

    pickle.dump(model.epoch_accuracy_list, open(graph_path+directory+'/Epoch_Accuracy_list.p','wb'))
    pickle.dump(model.epoch_validation_list, open(graph_path+directory+'/Epoch_Validation_list.p','wb'))
    pickle.dump(model.batch_loss_list, open(graph_path+directory+'/Batch_loss_list.p','wb'))
    pickle.dump(model.batch_accuracy_list, open(graph_path+directory+'/Batch_Accuracy_list.p','wb'))
    pickle.dump(model.confusion_m, open(graph_path+directory+'/Confusion_Matrix.p','wb'))

    model.save_weights(f'./TrainedModels/BinaryClassification/AlexNet/AlexNetMod{model.epoch_accuracy_list[-1]}-{model.epoch_validation_list[-1]}-{dt_string}.h5')

if __name__ == "__main__":
    main()

