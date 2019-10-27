import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Transforms import *
import numpy as np
import pdb

class YOLOLoss(nn.modules.loss._Loss):

    def __init__(self,numClasses, reduction = 32, anchors = [[1.3221 , 3.19275, 5.05587, 9.47112,11.2364],[1.73145, 4.00944, 8.09892, 4.94053,10.0071]],coordScale = 5 ,noObjScale = .5,objScale = 1, device = "cpu"):
        super(YOLOLoss, self).__init__()
        self.numOfClass = numClasses
        self.anchors = torch.tensor(anchors,dtype=torch.float).to(device)
        self.reduction = reduction
        self.coordScale = coordScale
        self.noObjScale = noObjScale
        self.objScale = objScale
        self.device = device

    def forward(self,output,target):
        # pdb.set_trace()
        bSize = output.shape[0]
        hSize = output.shape[1]
        wSize = output.shape[2] 

        #take parameters from output func        
        predConf = torch.sigmoid(output[:,:,:,:,4]) #predConf  = torch.zeros_like(output[:,:,:,:,4])      #conf = torch.sigmoid(output[:,:,:,:,4])
        predClass  = output[:,:,:,:,5:]

        predLoc = torch.zeros_like(output[:,:,:,:,0:4])
        predLoc[:,:,:,:,0:2] = torch.sigmoid(output[:,:,:,:,0:2])
        predLoc[:,:,:,:,2:4] = output[:,:,:,:,2:4]
        
        predTrueLoc = self.createPredCoord(hSize,wSize,output[:,:,:,:,0:4])

        # targetMonitor = target[0].clone()
        
        target,gridX,gridY,numOfElement = self.findLocOfTargets(target)
        selectedAnc = self.findBestAnchor(target)

        coordMask = torch.zeros_like(predLoc)
        confMask  = torch.ones_like(predConf)*self.noObjScale

        trueLoc =  torch.zeros_like(predLoc)
        trueConf = torch.zeros_like(predConf)

        maskedTrueClass = torch.ones((target.shape[0],1)).to(self.device)*-1
        maskedPredClass = torch.zeros((target.shape[0],self.numOfClass)).to(self.device)
        
        totalElNum = 0
        for it in range(bSize):
            tempElNum = numOfElement[it]
            tempGridX = gridX[totalElNum : totalElNum+tempElNum]
            tempGridY = gridY[totalElNum : totalElNum+tempElNum]
            tempSelAnc = selectedAnc[totalElNum : totalElNum+tempElNum]
            tempTarget = target[totalElNum : totalElNum+tempElNum,:]
            tempPredLoc = predTrueLoc[it,tempGridX,tempGridY,tempSelAnc,:]

            #calc iou between the true target and true pred loc 
           
            iou = self.calcIOU(tempPredLoc,tempTarget[:,1:])
            coordMask[it,tempGridX,tempGridY,tempSelAnc,:] = 1
            confMask[it,tempGridX,tempGridY,tempSelAnc] = self.objScale

            trueLoc[it,tempGridX,tempGridY,tempSelAnc,0] = tempTarget[:,1] - tempGridX.type(torch.float)
            trueLoc[it,tempGridX,tempGridY,tempSelAnc,1] = tempTarget[:,2] - tempGridY.type(torch.float)
            trueLoc[it,tempGridX,tempGridY,tempSelAnc,2] = torch.log(tempTarget[:,3] / self.anchors[0,tempSelAnc])
            trueLoc[it,tempGridX,tempGridY,tempSelAnc,3] = torch.log(tempTarget[:,4] / self.anchors[1,tempSelAnc])

            # axx = trueLoc[it,tempGridX,tempGridY,tempSelAnc,2]
            # bxx = trueLoc[it,tempGridX,tempGridY,tempSelAnc,3]
            # for ik in axx:
            #     if torch.isnan(ik):
            #         pdb.set_trace()

            # for ik in bxx:
            #     if torch.isnan(ik):
            #         pdb.set_trace()

            # change the predLoc and target location in the same resolution.                
            
            # print("trueLoc", trueLoc[it,tempGridX,tempGridY,tempSelAnc,:].detach().cpu())
            # print("predLoc",predLoc[it,tempGridX,tempGridY,tempSelAnc,:].detach().cpu())

            trueConf[it,tempGridX,tempGridY,tempSelAnc] = iou.diag()

            maskedTrueClass[totalElNum:totalElNum+tempElNum,:] = tempTarget[:,0].reshape((-1,1))
            maskedPredClass[totalElNum:totalElNum+tempElNum,:] = predClass[it,tempGridX,tempGridY,tempSelAnc,:]

            totalElNum += tempElNum

        # pdb.set_trace()
        MSE1 = nn.MSELoss(size_average=False).to(self.device)
        # MSE2 = nn.MSELoss()
        CENT = nn.CrossEntropyLoss(size_average=False).to(self.device)
        
        # pdb.set_trace()
        coordLoss = self.coordScale * MSE1(predLoc*coordMask, trueLoc*coordMask)/target.shape[0]
        # print("coordLoss",coordLoss.item())

        confLoss = MSE1(predConf*confMask, trueConf*confMask)/target.shape[0] # all scale applied in mask.
        maskedTrueClass = maskedTrueClass.reshape((-1)).type(torch.long)
        
        classLoss = CENT(maskedPredClass,maskedTrueClass)/target.shape[0]
        # print("pred",maskedPredClass.argmax(dim=1))
        # print("truth",maskedTrueClass)
        # print("loss",classLoss.item())
        
        lossAll = coordLoss + confLoss + classLoss 
        if torch.isnan(lossAll):
            pdb.set_trace()
        # lossAll = coordLoss 
        # print("lossAll",lossAll.item())
        # pdb.set_trace()
        return coordLoss,confLoss,classLoss,lossAll

    def findBestAnchor(self,target):
        anchorNum = self.anchors.shape[1]
        tempAnchor = torch.zeros((anchorNum,5)).to(self.device)
        tempAnchor[:,3:] = self.anchors.t()
        tempTarget = target.clone()
        tempTarget[:,1:3] = 0
        # pdb.set_trace()
        tempIOU = self.calcIOU(tempTarget[:,1:],tempAnchor[:,1:])
        _ ,selectedAnchor = tempIOU.max(1)
        return selectedAnchor.type(torch.long)

    def calcIOU(self,box1,box2):

        minX1,minY1 = (box1[:,0:2] - box1[:,2:]/2).split(1,1)
        maxX1,maxY1 = (box1[:,0:2] + box1[:,2:]/2).split(1,1)

        minX2,minY2= (box2[:,0:2] - box2[:,2:]/2).split(1,1)
        maxX2,maxY2 = (box2[:,0:2] + box2[:,2:]/2).split(1,1)

        dx = (maxX1.min(maxX2.t()) - minX1.max(minX2.t())).clamp(min=0)
        dy = (maxY1.min(maxY2.t()) - minY1.max(minY2.t())).clamp(min=0)
        interSec = dx*dy

        area1 = (maxX1 - minX1) * (maxY1-minY1)
        area2 = (maxX2 - minX2) * (maxY2-minY2)

        unions = (area1 + area2.t())- interSec
        IOU = interSec/unions
        
        return IOU

    def findLocOfTargets(self,target):
        numOfElement = target[1]
        target = self.createTargetLoc(target)
        gridX = target[:,1].type(torch.long)
        gridY = target[:,2].type(torch.long) 
        return target,gridX,gridY,numOfElement


    def createTargetLoc(self,target):
        bBoxes = target[0]
        tempBoxes = torch.zeros_like(bBoxes)
        tempBoxes[:,0] = bBoxes[:,0]
        tempBoxes[:,1] = (bBoxes[:,1] + (bBoxes[:,3]-bBoxes[:,1])/2)/self.reduction
        tempBoxes[:,2] = (bBoxes[:,2] + (bBoxes[:,4]-bBoxes[:,2])/2)/self.reduction
        tempBoxes[:,3] = (bBoxes[:,3] - bBoxes[:,1])/self.reduction
        tempBoxes[:,4] = (bBoxes[:,4] - bBoxes[:,2])/self.reduction
        return tempBoxes

    def createPredCoord(self,hSize,wSize,predLoc):
        # create start point of the input image.    
        tempPredLoc = torch.zeros_like(predLoc)
        tempPredLoc[:,:,:,:,0:2] = torch.sigmoid(predLoc[:,:,:,:,0:2])
        tempPredLoc[:,:,:,:,2:4] = torch.exp(predLoc[:,:,:,:,2:4])
        
        # gridX = torch.arange(wSize,requires_grad = False,dtype = torch.float).repeat(hSize,1)
        gridX = torch.arange(wSize,requires_grad = False,dtype = torch.float,device = self.device).reshape((wSize,1)).repeat(1,hSize)
        gridY = torch.arange(hSize,requires_grad = False,dtype = torch.float,device = self.device).reshape(1,hSize).repeat(hSize,1)
        tempLoc = torch.zeros_like(predLoc)

        tempLoc[:,:,:,:,0] = tempPredLoc[:,:,:,:,0] + gridX.reshape(1,hSize,wSize,1)
        tempLoc[:,:,:,:,1] = tempPredLoc[:,:,:,:,1] + gridY.reshape(1,hSize,wSize,1)
        tempLoc[:,:,:,:,2] = tempPredLoc[:,:,:,:,2] * self.anchors[0]
        tempLoc[:,:,:,:,3] = tempPredLoc[:,:,:,:,3] * self.anchors[1]

        return tempLoc

    def __call__(self,output,target):
        return self.forward(output,target)

    def monkeyLossFun(self,predLoc,gridX,gridY,selectedAnc,numOfElement,target):
        with torch.no_grad():
            dummyTarget = target[:,1:]
            # dummyTarget[:,2:4] = dummyTarget[:,2:4].sqrt()
            tempLoc = 0
            calcLoc = torch.FloatTensor([])
            for i,ele in enumerate(numOfElement):
                dummyLoc = predLoc[i,gridX[tempLoc:tempLoc+ele],gridY[tempLoc:tempLoc+ele],selectedAnc[tempLoc:tempLoc+ele]]
                calcLoc = torch.cat((calcLoc,dummyLoc))
                tempLoc += ele
            dummyLoss = calcLoc-dummyTarget
            lossCalc = dummyLoss*dummyLoss
            lossCalc = torch.sum(lossCalc)
        return lossCalc,calcLoc

if __name__ == "__main__":
    from Data import *
    GX = 13
    GY = 13
    numOfClass = 20
    batchSize = 4
    anchorNum = 5

    dataLoader = getData(416,batchSize=batchSize,sample = 10)
    loss= YOLOLoss(numOfClass)
    draw = DrawbBoxCenterBatch()
    for it,data in enumerate(dataLoader):
        # pdb.set_trace()
                # output = torch.arange(batchSize*GX*GY*anchorNum*(numOfClass+5),dtype = torch.float).reshape(batchSize,GX,GY,anchorNum,numOfClass+5)
        output = torch.zeros((batchSize,GX,GY,anchorNum,(numOfClass+5)),dtype = torch.float)
        target = (data["bBox"],data["numOfElement"])
        coordLoss,confLoss,classLoss,lossAll = loss.forward(output,target)
        # grid = (gridX,gridY)
        # draw(data,grid,str(it))
        print("-----------")

    # target = torch.arange(batchSize*25,dtype =torch.float).reshape(batchSize,numOfClass+5)
    # loss.forward(output,target)
    



