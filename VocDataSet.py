import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import xml.etree.ElementTree as elementT
import matplotlib.pyplot as plt

class csvRead():
    def __init__(self,fileName,path):
        self.fileName = fileName
        self.path = path

    def readCsv(self):
        fileDest = self.path+ "/ImageSets/Main/" + self.fileName
        data = pd.read_csv(fileDest, sep='\s+', header=-1, dtype='str')
        return data[0].values

    def __call__(self):
        return self.readCsv()


class VOCDataSet(Dataset):
    def __init__(self, folder, dataName, sample = -1, transform=None):
        """ Reading images with jpg suffix"""
        self.folder = folder
        self.imageName = dataName
        # np.random.shuffle(self.imageName)
        if sample != -1:
            self.imageName = self.imageName[:sample]

        self.classes = np.array(['sheep', 'horse', 'bicycle', 'bottle', 'cow', 'sofa', 'car', 'dog', 'cat', 'person', 'train', 'diningtable', 'aeroplane', 'bus', 'pottedplant', 'tvmonitor', 'chair', 'bird', 'boat', 'motorbike'])
        self.transform = transform

    def __len__(self):
        """len function for Dataset"""
        return len(self.imageName)

    def parseXml(self,xmlFile):
        """parse xml for bounding box and class name"""
        fid = open(xmlFile)
        tree = elementT.parse(fid)
        root = tree.getroot()
        imSize = root.find('size')
        width = int(imSize.find('width').text)
        height = int(imSize.find('height').text)

        allObj = []
        for obj in root.iter('object'):
            nameOfObj = obj.find('name').text
            # id = [it for it,name in enumerate(self.classes) if name == nameOfObj]
            id = np.argwhere(self.classes==nameOfObj)[0][0]
            bBox = obj.find('bndbox')
            xmin = int(bBox.find('xmin').text)+1
            xmax = int(bBox.find('xmax').text)-1
            ymin = int(bBox.find('ymin').text)+1
            ymax = int(bBox.find('ymax').text)-1
            allObj.append([id,xmin,ymin,xmax,ymax])
        fid.close()
        return np.array(allObj)

    def __getitem__(self,idx):
        """getitem for dataLoader"""
        # import pdb; pdb.set_trace()
        imageName = self.imageName[idx]
        # imageName = "2008_005090"
        image2Open = self.folder + "/JPEGImages/" + imageName+".jpg"
        dataFolder = self.folder + "/Annotations/" + imageName+".xml"
        boxData = self.parseXml(dataFolder)
        # print(image2Open)
        # print(dataFolder)
        imageData = np.array(cv2.cvtColor(cv2.imread(image2Open),cv2.COLOR_BGR2RGB))
        outData = {"imData":imageData,"bBox":boxData}
        if self.transform:
            outData = self.transform(outData)
        return outData

    # def __call__(self):
    #     tempFile = self.folder + "/Annotations/" + self.imageName[1] + ".xml"
    #     print(self.parseXml(tempFile))

if __name__=="__main__":
    csv = csvRead("train.txt","/home/osmant/VOCdevkit/VOC2007")
    data = csv()
    # print(len(data))

    voc = VOCDataSet("/home/osmant/VOCdevkit/VOC2007",data,sample=100)
    print(len(voc))
    outData = voc[1]
    print(outData["imData"].shape)
    plt.imshow(outData["imData"])
    plt.show()
    
