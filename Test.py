import torch
import torch.nn as nn
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from yoloVOC import classTinyArch as architecture

import argparse
from utils import *
from Transforms import testAdjustment,DrawBbox


parser = argparse.ArgumentParser(description="YOLO detection implementation")
parser.add_argument("-imPath", "--imPath", default="/Images", type=str, help="imageName for test")
parser.add_argument("-dLoad", "--detectLoad",default='/logs/2/BestDetectionData.pth', help="detection model to Load")
parser.add_argument("-imSize", "--imSize",default='416',type = int, help="imSize")
parser.add_argument("-classNum", "--numOfClass", default=20,type=int, help='class number in dataset,default = 20')
parser.add_argument("-anchorNum", "--anchorNum", default=5,type=int, help='anchor num for each grid, default = 5')
parser.add_argument("-gridNum", "--gridNum", default=13,type=int, help="grid in x,y dimension,default = 13")
parser.add_argument("-testNum", "--testNum", default = 10,type=int, help="testNum ,default = 3")
args = parser.parse_args()
args.directory = os.getcwd()
args.imPath = args.directory + args.imPath 
args.detectLoad =  args.directory + args.detectLoad

classes = np.array(['sheep', 'horse', 'bicycle', 'bottle', 'cow', 'sofa', 'car', 'dog', 'cat', 'person', 'train', 'diningtable', 'aeroplane', 'bus', 'pottedplant', 'tvmonitor', 'chair', 'bird', 'boat', 'motorbike'])           

_,net,_,_,_,_,_ = loadCheckpoint(args.detectLoad)
net = net.cpu()


testObj = testAdjustment(args.imSize,args.testNum)
drawObj = DrawBbox()
for it in range(args.testNum):
    imDict = testObj(it)
    outputData = net(imDict["imData"]).reshape((imDict["imData"].shape[0], args.gridNum, args.gridNum, args.anchorNum, -1)) 
    drawObj(imDict,"Orig")
    
    bBoxes = calcOutput(outputData)
    
    tempBoxes = torch.ones((20,5))*-5
    tempBoxes[:bBoxes.shape[0],1:] = bBoxes
    import pdb; pdb.set_trace()
    imDict["bBox"] = tempBoxes
    drawObj(imDict,"network")
    
    
    





