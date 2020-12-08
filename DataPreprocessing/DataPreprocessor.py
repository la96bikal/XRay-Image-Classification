#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 
import glob
import numpy as np

#Counters for Errors relating to YOLO crop
premature_size_counter = 0
uncropped_counter = 0

#Logging issues to a text file for problems while cropping images with YOLO
def file_writer(str_to_write, option):
    file1 = open("Problematic_images.txt", option)#append mode 
    file1.write(f"{str_to_write} \n") 
    file1.close() 

#DataPreprocessor that takes in the YOLO net, the dataset to process and the desired size of the image
def preprocess_data(net, dataset, image_size):
    global premature_size_counter
    global uncropped_counter

    pbar = tqdm(total = len(dataset))
    img_path = ''
    dataset.reset_index(inplace = True)
    try:    
        for x in range(len(dataset)):
            img_path = dataset.loc[x]["Path"]                        
            read_image = cv2.imread(img_path)    
            cv2.imwrite(f'{img_path[:-4]}_crop.jpg', resize_and_crop_img(net, read_image, image_size, img_path))
            pbar.update(1)
        pbar.close()

    except FileNotFoundError:
        print('Path Error.')
    except cv2.error as c:        
        file_writer(f'Uncropped,{img_path}', 'a')
        uncropped_counter += 1
        cv2.imwrite(f'{img_path[:-4]}_crop.jpg', cv2.resize(cv2.imread(img_path), image_size))
    except NameError as n:        
        print('EXCEPTION! Please verify paths for --> ', img_path)
    except Exception as e:
        print('An error occured '+ str(e))  
        print(type(e)) 

#Crops the image through YOLO and then resizes the images to the desired size using OPENCV
def resize_and_crop_img(net, img, size, img_path):
    global premature_size_counter
    global uncropped_counter

    cropped_img = yolo_cropper(net, img)        
    if (cropped_img.shape[0] < size[0] or cropped_img.shape[1] < size[1]):
        file_writer(f'Premature_size,{img_path}', 'a')
        premature_size_counter += 1  
        resized_img = cv2.resize(img, size)        
        return resized_img
    else:
        resized_img = cv2.resize(cropped_img, size)         
        return resized_img

#Codes for cropping the image based on YOLO Predictions
def yolo_cropper(net, img):   
    classes = ['xray']    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size = (len(classes), 3))
    
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)    

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    last_box = len(boxes) - 1
    x,y,w,h = boxes[last_box][0], boxes[last_box][1], boxes[last_box][2], boxes[last_box][3]
    
    # to make sure that the bounding box are not out of bound
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > img.shape[1]:
        w = img.shape[1] - x
    if y + h > img.shape[0]:
        h = img.shape[0] - y
    
    crop_img = img[y:y+h, x:x+w]   
    
    return crop_img

#The driver method
def main(): 
    try:
        global premature_size_counter
        global uncropped_counter

        premature_size_counter = 0
        uncropped_counter = 0

        file_writer('Start', 'w')
        image_size = (500,500)
        data = pd.read_csv('CheXphoto-v1.0/train.csv')        
        data["Synthetic/Natural"] = data["Path"].apply(lambda x: x.split("/")[2])

        data_natural = data[data["Synthetic/Natural"] == "natural"]

        net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
        preprocess_data(net, data_natural, image_size)

        file_writer(f'The total number of uncropped images are {uncropped_counter}', 'a')
        file_writer(f'The total number of premature sized images are {premature_size_counter}', 'a')

    except FileNotFoundError:
        print('EXCEPTION! Train.csv does not exist. Please check your directory.') 

    except cv2.error:
        print('EXCEPTION! Could not find the YOLO model. Please verify.')

    except Exception as inst:
        print('An error occured '+ str(inst)) 
        print(type(inst))


if __name__ == "__main__":
    main()



