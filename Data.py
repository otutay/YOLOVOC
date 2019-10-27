from VocDataSet import *
from Transforms import *
import torch
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,ConcatDataset
import pdb
from torch.utils.data.dataloader import default_collate
def collateDict(imDict): 
    imDict = np.array(imDict)
    tempLoc = imDict != {}
    tempImDict = imDict[tempLoc]
    dataDict = default_collate(tempImDict)
    return dataDict


def getData(imSize,file2Read = "train.txt", path = "/home/osmant/VOCdevkit/",batchSize = 10,sample = -1):
    voc2007Path = path + "VOC2007"
    voc2012Path = path + "VOC2012"
    # pdb.set_trace()
    csv2007 = csvRead(file2Read,voc2007Path)
    csv2012 = csvRead(file2Read,voc2012Path)

    data2Read2007 = csv2007()
    data2Read2012 = csv2012()

    transformsFn = transforms.Compose([RandomCrop(),RandomFlip(),ReScaleData((imSize,imSize)),NormalizeData(),EliminateSmallBoxes(0.0015),ToTensor()])

    data2007 = VOCDataSet(voc2007Path,data2Read2007,sample = sample,transform=transformsFn)
    data2012 = VOCDataSet(voc2012Path,data2Read2012,sample = sample,transform=transformsFn)
    
    dataLoader = DataLoader(ConcatDataset([data2007, data2012]),batch_size= batchSize,shuffle = True,collate_fn = collateDict,num_workers = 0,pin_memory=True)

    return dataLoader

if __name__=="__main__":
    dataLoader = getData(416,sample = 100)
    draw = DrawBbox()
    for it,data in enumerate(dataLoader):
        pdb.set_trace()
        print(it)
        numOfPic = data["imData"].shape
        print(numOfPic[0])
        dataDict = {}
        totalBBox = 0
        for ik in range(numOfPic[0]):
            dataDict = {}
            numOfBbox = data["numOfElement"][ik]
            dataDict["imData"] = np.transpose(data["imData"][ik].numpy(),(1,2,0))
            dataDict["bBox"] = data["bBox"][totalBBox:numOfBbox+totalBBox].numpy()
            totalBBox += numOfBbox
            draw(dataDict,str(it*numOfPic[0]+ik))
    



