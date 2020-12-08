import pandas as pd
import numpy as np
import os 
import cv2
import copy
from tqdm import tqdm
import sys
import pickle
import argparse
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def remove_negative_rows(data, targets):
    data = data[(data[targets[0]] != -1) & 
                (data[targets[1]] != -1) &
                (data[targets[2]] != -1) &
                (data[targets[3]] != -1) &
                (data[targets[4]] != -1)]
    return data

def data_retriever(data, targets):
    data["Synthetic/Natural"] = data["Path"].apply(lambda x: x.split("/")[2])
#     data_synthetic = data[data["Synthetic/Natural"] == "synthetic"]
    data_synthetic = data[data["Synthetic/Natural"] == "synthetic"]
    data_synthetic = data_synthetic.fillna(0) #Filling Nans with 0 
    data_synthetic = remove_negative_rows(data_synthetic, targets)
    data_synthetic.reset_index(inplace = True) 

#     data_natural = data[data["Synthetic/Natural"] == "natural"]
    data_natural = data[data["Synthetic/Natural"] == "natural"]
    data_natural = data_natural.fillna(0) #Filling Nans with 0
    data_natural = remove_negative_rows(data_natural, targets)
    data_natural.reset_index(inplace = True)    
    return data_natural, data_synthetic

def extract_and_resize(data, path_to_data, natural_or_synthetic, image_size, targets):
    dataset = []
    positions = {"AP": [1, 0, 0], "PA": [0, 1, 0], 0: [0, 0, 1]}
    if natural_or_synthetic == "natural":
        print("Extracting and resizing Natural Data")  
        pbar = tqdm(total = len(data))
        for x in range(len(data)):
            # Creating tuple of the cropped image and the target value
            try:
                dataset.append((cv2.resize(cv2.imread((os.path.join(path_to_data, data.loc[x]["Path"][:-4] + ".jpg"))), image_size), 
                                [data.loc[x][target] for target in targets], 
                                data.loc[x]["Sex"], 
                                data.loc[x]["Age"],
                                positions[data.loc[x]["AP/PA"]]))
            except:
                continue
            pbar.update(1)
        pbar.close()        
        
    elif natural_or_synthetic == "synthetic":        
        print("Extracting and resizing Synthetic Data")
        pbar = tqdm(total = len(data))
        for x in range(len(data)):
            # Creating tuple of the synthetic image and the target value
            try:
                dataset.append((cv2.resize(cv2.imread((os.path.join(path_to_data, data.loc[x]["Path"]))), image_size), 
                                [data.loc[x][target] for target in targets], 
                                data.loc[x]["Sex"], 
                                data.loc[x]["Age"],
                                positions[data.loc[x]["AP/PA"]]))
            except:
                continue
            pbar.update(1)
        pbar.close()    
    else:
        print("Invalid Selection for Natural or Synthetic")
        return None
    return dataset

def tuple_dataset(data, path_to_data, natural_or_synthetic, image_size, targets):
    data_natural, data_synthetic = data_retriever(data, targets)
    dataset = []
    if natural_or_synthetic.lower() == "natural":
        dataset.extend(extract_and_resize(data_natural, path_to_data, "natural", image_size, targets))
    elif natural_or_synthetic.lower() == "synthetic":
        dataset.extend(extract_and_resize(data_synthetic, path_to_data, "synthetic", image_size, targets))
    elif natural_or_synthetic.lower() == "both":
        dataset.extend(extract_and_resize(data_natural, path_to_data, "natural", image_size, targets))
        dataset.extend(extract_and_resize(data_synthetic, path_to_data, "synthetic", image_size, targets))
    else:
        print("Invalid Selection for natural or synthetic")
        return None    	
    return dataset

def balance_dataset(dataset, aug_num, label):
    old_data = np.array(dataset)
    if label == -1:
        min_images = [pair for pair in old_data if sum(pair[1]) == 0]
    elif label <= 4 and label >= 0:
        if label == 2:
            min_images = [pair for pair in old_data if pair[1][label] == 1]
        else:
            min_images = [pair for pair in old_data if (pair[1][label] == 1 and pair[1][2] != 1)]
    else:
        print("Check Label")
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
    
    print(f"Generating and Balancing dataset...")
    total = 0
    augmented_data = []
    pbar = tqdm(total = aug_num)
    while total <= aug_num:
        total += 1
        min_image = copy.deepcopy(random.choice(min_images))
        temp_image = min_image[0]
        temp_image = temp_image.reshape(1, temp_image.shape[0], temp_image.shape[1], temp_image.shape[2])
        imageGen = aug.flow(temp_image, batch_size=1)
        min_image[0] = imageGen[0][0]
        augmented_data.append(min_image)
        pbar.update(1)
    pbar.close()
    
    new_data = []
    new_data.extend(list(old_data))
    
    for item in augmented_data:
        new_data.append(item)
    new_data = np.array(new_data)
    np.random.shuffle(new_data)
    return new_data

def data_info(data, targets):
    labels = [pair[1] for pair in data]
    labels_ind = [[label[0] for label in labels],
                 [label[1] for label in labels],
                 [label[2] for label in labels],
                 [label[3] for label in labels],
                 [label[4] for label in labels]]
    nums = [sum(label) for label in labels_ind]
    temp = [sum(label) for label in labels]
    neg_num = len([y for y in temp if y == 0])

    label_num = {}
    for num, label in zip(nums, targets):
        label_num[label] = num
    label_num["no_finding"] = neg_num
    return label_num


def main():
    # parameterizing our script. You can also run the default values to replicate our result. 
    parser = argparse.ArgumentParser(description='CheXphoto Data Extractor for multilable classification')
    parser.add_argument('--path_to_data', default='..', type=str, help='Path to the folder where CheXphoto folder lies.Defualt .')
    parser.add_argument('--data_type', default='both', type=str, help='synthetic or natural or both')
    parser.add_argument('--image_size', default='224x224', type=str, help="Shape of the image to resize to. Defualt is 224x224")
    parser.add_argument('--aug_num', default='3000#2000#0#4000#0#0', type=str, help='Provide Augmentation number for each classes. The last number is reserved for no finding. The order of positive labels is same as below.')
    parser.add_argument('--load_data', default='False', type=str, help="If you have already ran this script once, False. If you simply wants to load the saved data, True")
    parser.add_argument('--balance', default='False', type=str, help="True if you want to run augmentation, or else False. Defualt True")
    arguments = parser.parse_args()

    parameters = {'targets' : ["Cardiomegaly", "Atelectasis", "Pleural Effusion", "Consolidation","Edema"], 'synthetic_or_natural': ['synthetic', 'natural', 'both']}           
    path_to_data = arguments.path_to_data            
    synthetic_or_natural = arguments.data_type.lower()
    image_size = arguments.image_size.split('x') #in format axb for ex: 227x227
    aug_num = arguments.aug_num.split("#")
    load_data = arguments.load_data .lower()
    balance = arguments.balance.lower()

    data = pd.read_csv(os.path.join(path_to_data, 'CheXphoto-v1.0/train.csv'))

    assert synthetic_or_natural.lower() in parameters['synthetic_or_natural'], "Invalid synthetic or natural value" 
        
    if load_data == "true":
        data = pickle.load(open(f"./Data/Complete_{image_size[0]}x{image_size[1]}_{synthetic_or_natural}_unbalanced.p","rb"))
    else:
        data = tuple_dataset(data, path_to_data, synthetic_or_natural, (int(image_size[0]), int(image_size[1])), parameters["targets"])
        pickle.dump(data, open(f"./Data/Complete_{image_size[0]}x{image_size[1]}_{synthetic_or_natural}_unbalanced.p", "wb"))
    print("Before Augmentation: ")
    print(data_info(data, parameters["targets"]))

    if balance == "false":
        exit()

    new_data = copy.deepcopy(data)
    new_data = balance_dataset(new_data, int(aug_num[0]), 0)
    new_data = balance_dataset(new_data, int(aug_num[1]), 1)
    new_data = balance_dataset(new_data, int(aug_num[2]), 2)
    new_data = balance_dataset(new_data, int(aug_num[3]), 3)
    new_data = balance_dataset(new_data, int(aug_num[4]), 4)
    new_data = balance_dataset(new_data, int(aug_num[5]), -1)

    print("After Augmentation")
    print(data_info(new_data, parameters["targets"]))
    pickle.dump(new_data, open(f"./Data/Complete_{image_size[0]}x{image_size[1]}_{synthetic_or_natural}_balanced.p", "wb"))

if __name__ == "__main__":
    main()