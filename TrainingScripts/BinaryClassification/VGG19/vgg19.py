import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tqdm import tqdm
import argparse
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import numpy as np

class VGG19(tf.keras.Model):
    
    def __init__(self, num_classes, image_size, optimizer):
        super(VGG19, self).__init__()
        
        self.input_size = image_size
        self.optimizer = optimizer

        #We use this list for graphing loss per batch
        self.loss_list = []
        
       #Convolutional layers
    
        self.conv1_1 = Conv2D(filters = 64, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act1_1 = Activation('relu')        
        

        self.conv1_2 = Conv2D(filters = 64, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act1_2 = Activation('relu') 
        
        #Pool
        self.pool_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2))
       
        self.conv2_1 = Conv2D(filters = 128, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act2_1 = Activation('relu')        
        

        self.conv2_2 = Conv2D(filters = 128, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act2_2 = Activation('relu')   
        
        #Pooling
        self.pool_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2))
        
        
        self.conv3_1 = Conv2D(filters = 256, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act3_1 = Activation('relu')        
       
        
        self.conv3_2 = Conv2D(filters = 256, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act3_2 = Activation('relu')        
        
        
        
        self.conv3_3 = Conv2D(filters = 256, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act3_3= Activation('relu')        
       
        
        self.conv3_4 = Conv2D(filters = 256, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act3_4 = Activation('relu')
        
        #Pooling
        self.pool_3 = MaxPooling2D(pool_size=(2,2), strides = (2,2))
        
        self.conv4_1 = Conv2D(filters = 512, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act4_1 = Activation('relu')        
        
        
        self.conv4_2 = Conv2D(filters = 512, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act4_2 = Activation('relu')        
        
        
        
        self.conv4_3 = Conv2D(filters = 512, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act4_3 = Activation('relu')        
         
        
        self.conv4_4 = Conv2D(filters = 512, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act4_4 = Activation('relu')   
        
        #Pooling
        self.pool_4 = MaxPooling2D(pool_size=(2,2), strides = (2,2))
        
        
        self.conv5_1 = Conv2D(filters = 512, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act5_1 = Activation('relu')        
        
        
        self.conv5_2 = Conv2D(filters = 512, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act5_2 = Activation('relu')        
       
        
        self.conv5_3 = Conv2D(filters = 512, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act5_3 = Activation('relu')        
        
        
        self.conv5_4 = Conv2D(filters = 512, input_shape = image_size, kernel_size = (3,3), strides = (1,1), padding = 'same')
        self.act5_4 = Activation('relu')  
        
        #Pooling
        self.pool_5 = MaxPooling2D(pool_size=(2,2), strides = (2,2))
       

        self.flatten = Flatten()
        
        #Fully connected layers
        self.fc1 = Dense(4096, input_shape = (224*224*3,))
        self.act1 = Activation('relu')
        
            
        self.fc2 = Dense(4096)
        self.act2 = Activation('relu')
        

        self.fc3 = Dense(num_classes)
        self.output_final = Activation('softmax')
        
    def call(self, inputs):
        
        
        
        layer1 = self.conv1_1(inputs)
        layer1 = self.act1_1(layer1)
       
        
        layer2 = self.conv1_2(layer1)
        layer2 = self.act1_2(layer2)
        layer2 = self.pool_1(layer2)
        
        layer3 = self.conv2_1(layer2)
        layer3 = self.act2_1(layer3)
        
        layer4 = self.conv2_2(layer3)
        layer4 = self.act2_2(layer4)
        layer4= self.pool_2(layer4)
        
        layer5 = self.conv3_1(layer4)
        layer5 = self.act3_1(layer5)
        
        layer6 = self.conv3_2(layer5)
        layer6= self.act3_2(layer6)

        layer7= self.conv3_3(layer6)
        layer7= self.act3_3(layer7)

        layer8= self.conv3_4(layer7)
        layer8= self.act3_4(layer8)
        layer8= self.pool_3(layer8)
        
        layer9 = self.conv4_1(layer8)
        layer9= self.act4_1(layer9)

        layer10 = self.conv4_2(layer9)
        layer10= self.act4_2(layer10)

        layer11 = self.conv4_3(layer10)
        layer11= self.act4_3(layer11)

        layer12 = self.conv4_4(layer11)
        layer12= self.act4_4(layer12)
        layer12= self.pool_4(layer12)

        layer13 = self.conv5_1(layer12)
        layer13 = self.act5_1(layer13)
        
        layer14 = self.conv5_2(layer13)
        layer14= self.act5_2(layer14)

        layer15= self.conv5_3(layer14)
        layer15= self.act5_3(layer15)

        layer16= self.conv5_4(layer15)
        layer16= self.act5_4(layer16)
        layer16= self.pool_5(layer16)

        flattened_layer = self.flatten(layer16)
        
        layer17 = self.fc1(flattened_layer)
        layer17 = self.act1(layer17)
        
        
        layer18 = self.fc2(layer17)
        layer18 = self.act2(layer18)
       
        
        layer19 = self.fc3(layer18)
        layer19 = self.output_final(layer19)
        
        return layer19   
    
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

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.1)
    dataset = []
    y_train =to_categorical(y_train)
    y_test =to_categorical(y_test)

    X_train = np.array(X_train, dtype = np.float32) / 255.0
    X_test = np.array(X_test, dtype = np.float32) / 255.0     

    return X_train, X_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser(description='VGG19 Binary Model')
    parser.add_argument('--path_to_data', default='./Data/Pleural Effusion_224x224_both_augmented.p', type=str, help="Pickle file location.")
    parser.add_argument('--epochs', default=20, type=int, help="Epochs for training.")
    parser.add_argument('--train_test_split', default='0.1', type=str, help='The parameter splits the full training data into train and test.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--input_shape', default='224x224x3', type=str, help="Input shape.")

    arguments = parser.parse_args()
    split = arguments.train_test_split
    epochs = arguments.epochs
    batch_size = arguments.batch_size
    lr = arguments.lr
    path_to_data = arguments.path_to_data
    image_size = arguments.input_shape

    dataset = pickle.load(open(path_to_data,'rb'))
    X_train, X_test, y_train, y_test = train_test_splitter(dataset, float(split))

    model = VGG19(2, image_size, optimizer = tf.keras.optimizers.Adam(learning_rate = lr))

    model.train(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_set = (X_test, y_test))

    # model.save_weights('VGG1985-78.h5')


if __name__ == "__main__":
    main()

