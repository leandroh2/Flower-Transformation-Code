#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:37:42 2019

@author: leandrohirai
"""

import os
import numpy as np
from shutil import copyfile

path = os.path.dirname(os.path.realpath(__file__))
#path = path+'/data/daySegData/JPEGImages'
path = path+'/data/day2nightData/JPEGImages'
txtPath = os.path.dirname(os.path.realpath(__file__))
#txtPath = txtPath+'/data/daySegData/ImageSets/Segmentation'
txtPath = txtPath+'/data/day2nightData/ImageSets/Segmentation'

imageNames = []
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file or '.jpg' in file:
            imageNames.append(file)
            
datasetSize = len(imageNames)
validation_split = 0.3
indices = list(range(datasetSize))
split = int(np.floor(validation_split*datasetSize))
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

trainNames = [imageNames[i] for i in train_indices]
valNames =[imageNames[i] for i in val_indices]
    
train_txt = open(txtPath+'/train.txt',"w")
val_txt = open(txtPath+'/val.txt',"w")
trainval_txt = open(txtPath+'/trainval.txt',"w")

for i in range(0,len(trainNames)):
    train_txt.write(trainNames[i]+'\n')
    trainval_txt.write(trainNames[i]+'\n')
    
for i in range(0, len(valNames)):
    val_txt.write(valNames[i]+'\n')
    trainval_txt.write(valNames[i]+'\n')
    
train_txt.close()
val_txt.close()
trainval_txt.close()