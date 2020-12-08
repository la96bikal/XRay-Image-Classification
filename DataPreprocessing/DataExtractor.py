import pandas as pd
import numpy as np
import os 
import cv2
from tqdm import tqdm
import sys
import pickle
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# Input Params: 
# data --> The Original dataset train.csv
# target --> The required target .. For ex: "Pleural Effusion", "Edema", etc 

# Returns: A tuple of original dataset filtered by natural and synthetic flags.

def data_retriever(data, target):
    data["Synthetic/Natural"] = data["Path"].apply(lambda x: x.split("/")[2])    
    data_synthetic = data[data["Synthetic/Natural"] == "synthetic"][data["Frontal/Lateral"] == "Frontal"]
    data_synthetic[target] = data_synthetic[target].fillna(0) #Filling Nans with 0 
    data_synthetic = data_synthetic[data_synthetic[target] >= 0] #Filtering -1s
    data_synthetic.reset_index(inplace = True) 

    data_natural = data[data["Synthetic/Natural"] == "natural"][data["Frontal/Lateral"] == "Frontal"]
    data_natural[target] = data_natural[target].fillna(0) #Filling Nans with 0
    data_natural = data_natural[data_natural[target] >= 0] #Filtering -1s
    data_natural.reset_index(inplace = True)    
    return data_natural, data_synthetic


#Input Params:
# data --> The filtered dataset from the data_retriever method where we filtered for natural/synthetic and the required target
# natural_or_synthetic --> Retrieve natural or Synthetic Data
# image_size --> Required size of the image for different models.
# target --> The required target .. For ex: "Pleural Effusion", "Edema", etc 

# The option for Natural or synthetic was given for this method for situations where we would only want to train for either 
# synthetic or natural data

# Retruns --> Natural or synthetic dataset in tuples where the first element of the tuple is the resized image and the second element is the target.
def extract_and_resize(data, path_to_data, natural_or_synthetic, image_size, target):    
    dataset = []
    if natural_or_synthetic == "natural":
        print("Extracting and resizing Natural Data")          
        pbar = tqdm(total = len(data))
        for x in range(len(data)):
            # Creating tuple of the cropped image and the target value
            try:
                target_value = data.loc[x][target]
                resized_img = (cv2.resize(cv2.imread((os.path.join(path_to_data, data.loc[x]["Path"][:-4] + "_crop.jpg"))), image_size))
                sex = 1 if data.loc[x]['Sex'] == 'Female' else 0
                age = data.loc[x]['Age']
                # frontal_or_lateral = 1 if data.loc[x]['Frontal/Lateral'] == 'Frontal' else 0
                dataset.append((resized_img, sex, age, target_value))
            except Exception as inst:
                print('Exception occured ::', str(inst))
                continue
            pbar.update(1)
        pbar.close()        
        
    elif natural_or_synthetic == "synthetic":        
        print("Extracting and resizing Synthetic Data")
        pbar = tqdm(total = len(data))
        for x in range(len(data)):
            # Creating tuple of the synthetic image and the target value
            try:
                target_value = data.loc[x][target]                
                resized_img = (cv2.resize(cv2.imread((os.path.join(path_to_data, data.loc[x]["Path"]))), image_size))
                sex = 1 if data.loc[x]['Sex'] == 'Female' else 0
                age = data.loc[x]['Age']
                # frontal_or_lateral = 1 if data.loc[x]['Frontal/Lateral'] == 'Frontal' else 0
                dataset.append((resized_img, sex, age, target_value))
            except Exception as inst: 
                print('Exception occured ::', str(inst))
                continue                
            pbar.update(1)
        pbar.close()    
    else:
        print("Invalid Selection for Natural or Synthetic")
        return None
    return dataset


# Input Params
# dataset : The complete tuple_dataset
#Returns: The balanced dataset with predefined augmentation

def balance_dataset(dataset):       
    old_data = dataset.copy()
    images, sex, age, label = zip(*dataset)  
    aug_num = 0  
    
    np_label = np.asarray(label)

    total = np_label.shape[0]
    positive_classes = np.sum(np_label)
    negative_classes = total - positive_classes
    minority_class = None

    if positive_classes > negative_classes:
        minority_class = 0
        aug_num = positive_classes - negative_classes        
    else:
        minority_class = 1
        aug_num = negative_classes - positive_classes

    min_image_tuples = [(images[i], sex[i], age[i]) for i in range(len(dataset)) if int(label[i]) == minority_class]       
    # min_images, sexes, ages = zip(*min_image_tuples)
    # min_images = np.array(min_images)

    aug_num = int(aug_num)
    aug = ImageDataGenerator(
        featurewise_center=False,
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
    
    print("Generating and Balancing dataset...")    
    total = 0
    augmented_data = []    
    pbar = tqdm(total = aug_num)

    for i in range(aug_num):      
        random_image = random.choice(min_image_tuples)            
        np_image = np.expand_dims(np.array(random_image[0]), axis=0)
        aug_image = aug.flow(np_image, batch_size = 1)
        augmented_data.append((aug_image[0][0],random_image[1] , random_image[2] , minority_class))
        if total == aug_num:
            break
        total += 1
        pbar.update(1)
    pbar.close()
    
    new_data = []
    new_data.extend(list(old_data))
    new_data.extend(augmented_data)

    new_data = np.array(new_data)
    np.random.shuffle(new_data)
    return new_data


# Input Params
# data : The original dataset (In our case train.csv)
# image_size : the desired size for the image
# target: The desired target
# natural_or_synthetic: Possible values for this parameter are "BOTH", "NATURAL", or "SYNTHETIC"

#Returns: The dataset in form of a tuple for both synthetic and 

def tuple_dataset(data, path_to_data, natural_or_synthetic, image_size, target):
    data_natural, data_synthetic = data_retriever(data, target)
    dataset = []
    if natural_or_synthetic.lower() == "natural":
        dataset.extend(extract_and_resize(data_natural, path_to_data, "natural", image_size, target))
    elif natural_or_synthetic.lower() == "synthetic":
        dataset.extend(extract_and_resize(data_synthetic, path_to_data, "synthetic", image_size, target))
    elif natural_or_synthetic.lower() == "both":
        dataset.extend(extract_and_resize(data_natural, path_to_data, "natural", image_size, target))
        dataset.extend(extract_and_resize(data_synthetic, path_to_data, "synthetic", image_size, target))
    else:
        print("Invalid Selection for natural or synthetic")
        return None     
    return dataset

#The driver method
# Input params: takes in path_to_data synthetic_or_natural parameter, image_size, target
# Example python DataPreprocessing\DataExtractor.py --path_to_data ..\.. --target Consolidation --data_type both --image_size 224x224 
def main():

    # parameterizing our script. You can also run the default values to replicate our result. 
    parser = argparse.ArgumentParser(description='CheXphoto Data Extractor')
    parser.add_argument('--path_to_data', default='.', type=str, help='Path to the folder where CheXphoto folder lies.Defualt .')
    parser.add_argument('--data_type', default='natural', type=str, help='synthetic or natural or both')
    parser.add_argument('--image_size', default='224x224', type=str, help="Shape of the image to resize to. Defualt is 224x224")
    parser.add_argument('--target', default='Pleural_Effusion', type=str, help='Class to separate. Default Pleural_Effusion')

    arguments = parser.parse_args()

    try:        
        parameters = {'targets' : ["Edema", "Pleural Effusion",  "Consolidation", "Cardiomegaly", "Atelectasis"], 'synthetic_or_natural': ['synthetic', 'natural', 'both']}           
        path_to_data = arguments.path_to_data            
        synthetic_or_natural = arguments.data_type
        image_size = arguments.image_size.split('x') #in format axb for ex: 227x227
        target = arguments.target
        target = target.replace("_", " ")    

        data = pd.read_csv(os.path.join(path_to_data, 'CheXphoto-v1.0/train.csv'))

        assert synthetic_or_natural.lower() in parameters['synthetic_or_natural'], "Invalid synthetic or natural value" 
        assert target in parameters['targets'], "Invalid target value"
          
        
        balanced_data = balance_dataset(tuple_dataset(data, path_to_data, synthetic_or_natural, (int(image_size[0]), int(image_size[1])), target))

        pickle.dump(balanced_data, open(f"./Data/{target}_{image_size[0]}x{image_size[1]}_{synthetic_or_natural}_augmented.p", "wb"))   

        print("Successfully dumped pickle files")   

    except FileNotFoundError:
        print('EXCEPTION! Train.csv does not exist. Please check your directory.') 

    except Exception as inst:
        print('An error occured '+ str(inst)) 
        print(type(inst))

if __name__ == "__main__":
    main()




