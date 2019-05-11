import cv2
import numpy as np
import os         
from random import shuffle     
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm 

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 50

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name
    
    if word_label == 'F':
        return np.array([1,0,0])
    elif word_label == 'R':
        return np.array([0,0,1])
    elif word_label == 'L':
        return np.array([0,1,0])
   
    
    
def create_train_data():
    training_data = []
    TRAIN_DIR = './F-S'
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        if 'jpg' in path:
            print(path)
            #img_data = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img_data = cv2.imread(path)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            #img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img_data), create_label("F")])
        
    TRAIN_DIR = './L-S'
    
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        
        if 'jpg' in path:
            print(path)
            #img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.imread(path)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
           
            #img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img_data), create_label("L")])
            
        
        
    TRAIN_DIR = './R-S'
    
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        if 'jpg' in path:
            print(path)
            #img_data = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img_data = cv2.imread(path)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            #img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img_data), create_label("R")])
        
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

data = create_train_data()