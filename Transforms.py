import numpy as np
import cv2
from VocDataSet import VOCDataSet
from VocDataSet import csvRead

import matplotlib.pyplot as plt
import torch
import pdb
class RandomCrop(object):
    def affineTransform(self,im,bBox):
        height,width,color = im.shape
        scale = np.random.rand()/10+1
        
        maxOffX = (scale-1)*width
        maxOffY = (scale-1)*height

        offX = int(np.random.rand()*maxOffX)
        offY = int(np.random.rand()*maxOffY)

        im = cv2.resize(im, (0,0), fx=scale, fy = scale)
        im = im[offY:(offY+height),offX:(offX+width)]

        offs = np.array([offX,offY]).reshape(1,2)
        offs = np.tile(offs,(bBox.shape[0],2))

        bBox[:,1:] = np.array(bBox[:,1:]*scale-offs,np.int32)
        bBox[:,1:] = self.interControl(bBox[:,1:],height,width)

        return im,bBox

    def interControl(self,coord,height,width):
        minLimit = [1,1,1,1]
        maxLimit = [width,height,width,height]
        
        minLimits = np.tile(minLimit,(coord.shape[0],1))
        maxLimits = np.tile(maxLimit,(coord.shape[0],1))
        
        compLower = coord < minLimits
        compHigher = coord > maxLimits
        
        coord[compLower] = minLimits[compLower]
        coord[compHigher] = maxLimits[compHigher]

        return coord

    def __call__(self,imDict):
        image1, bBox1 = imDict["imData"],imDict["bBox"]
        image,bBox = self.affineTransform(image1,bBox1)
        imDict["imData"]=image
        imDict["bBox"]=bBox
        return imDict

class RandomFlip(object):
    def doRandomFlip(self, im, bBox):
        temp =  np.random.rand()
        flipLR = temp>0.66
        flipUD = temp>0.33
        h,w,c = np.shape(im)

        if flipLR:
            im = cv2.flip(im,1)
            tempXmin = w-bBox[:,3]
            tempXmax = w-bBox[:,1]
            bBox[:,1] = tempXmin
            bBox[:,3] = tempXmax
        elif flipUD:
            im = cv2.flip(im,0)
            tempYmin = h-bBox[:,4]
            tempYmax = h-bBox[:,2]
            bBox[:,2] = tempYmin
            bBox[:,4] = tempYmax
        return im,bBox

    def __call__(self,imDict):
        im, bBox = self.doRandomFlip(imDict["imData"],imDict["bBox"])
        imDict["imData"] = im
        imDict["bBox"] = bBox
        return imDict

class ReScaleData(object):
    def __init__(self,size):
        self.width2Scale  = size[0]
        self.height2Scale = size[1]

    def __call__(self,imDict):
        im, bBox = imDict["imData"],imDict["bBox"]
        he,wid,col = np.shape(im)
        im = cv2.resize(im,(self.width2Scale,self.height2Scale))

        scaleW = self.width2Scale/wid
        scaleH = self.height2Scale/he
        bBox = np.array(bBox,np.float32)
        bBox[:,1] *= scaleW
        bBox[:,3] *= scaleW
        bBox[:,2] *= scaleH
        bBox[:,4] *= scaleH

        bBox[:,1:] = self.interControl(bBox[:,1:],self.height2Scale,self.width2Scale)
        imDict["imData"] = im
        imDict["bBox"] = bBox
        return imDict

    def interControl(self,coord,height,width):
        minLimit = [1,1,1,1]
        maxLimit = [width,height,width,height]
        
        minLimits = np.tile(minLimit,(coord.shape[0],1))
        maxLimits = np.tile(maxLimit,(coord.shape[0],1))
        
        compLower = coord < minLimits
        compHigher = coord > maxLimits
        
        coord[compLower] = minLimits[compLower]
        coord[compHigher] = maxLimits[compHigher]

        return coord

class NormalizeData(object):
    def __call__(self,imDict):
        im = np.array(imDict["imData"],np.float)        
        im /= 255.0
        imDict["imData"] = im
        return imDict

class EliminateSmallBoxes(object):
    def __init__(self, thresh):
        self.thresh = thresh

    def maskArea(self,bBox,allArea):
        width = bBox[:,3]-bBox[:,1]
        height = bBox[:,4]-bBox[:,2]
        area = width*height/allArea
        compVec = area>self.thresh
        bBox = bBox[compVec]
        return bBox

    def __call__(self, imDict):
        bBox = imDict['bBox']
        im = imDict["imData"]
        h,w,_ = np.shape(im)
        allArea = h*w
        bBox = self.maskArea(bBox,allArea)
        imDict['bBox'] = bBox

        return imDict


class ToTensor(object):
    def __init__(self,maxbBoxNum = 40):
        self.maxbBoxNum = maxbBoxNum

    def __call__(self,imDict):
        im = imDict["imData"]
        bBox = imDict["bBox"]
        im = np.transpose(im,(2,0,1))

        if len(bBox) == 0:
            return {}
        else:
            imDict["imData"] = torch.from_numpy(im).type(torch.float)
            imDict["bBox"] = torch.ones((1,5*self.maxbBoxNum))*-5
            imDict["bBox"][0,:5*len(bBox)] = torch.from_numpy(bBox.reshape(-1))
            imDict["numOfElement"] = torch.tensor(len(bBox)).reshape((1,-1))    
        return imDict

class testAdjustment():
    def __init__(self,imSize,testNum):
        self.tensorObj = ToTensor()
        self.rescaleObj = ReScaleData((imSize,imSize))
        self.normObj = NormalizeData()
        self.csvObj = csvRead("train.txt","/home/osmant/VOCdevkit/VOC2007")
        self.data = self.csvObj()
        self.vocDataSet = VOCDataSet("/home/osmant/VOCdevkit/VOC2007",self.data,sample=testNum)

    def __call__(self,id):
        imDict = self.vocDataSet[id]
    
        imDict = self.rescaleObj(imDict)
        imDict = self.normObj(imDict)

        imDict = self.tensorObj(imDict)
        imDict = self.addBatchDimen(imDict)
        return imDict

    def addBatchDimen(self,imDict):
        imDict["imData"] = imDict["imData"][None,:]
        return imDict

    
class DrawBbox(object):
    def drawImage(self,imDict,name):
        image = imDict["imData"].squeeze().permute((1,2,0))
        bBoxes  = imDict["bBox"]
        bBoxes = self.correctbBox(bBoxes)
        colors = ["red","blue","green","yellow","black","brown"]
        plt.figure()
        plt.imshow(image.numpy())
        # pdb.set_trace()
        for it,bBox in enumerate(bBoxes):
            xMin = bBox[1]
            xMax = bBox[3]
            yMin = bBox[2]
            yMax = bBox[4]
            plt.plot((xMin,xMin),(yMin,yMax),(xMax,xMax),(yMin,yMax),(xMin,xMax),(yMin,yMin),(xMin,xMax),(yMax,yMax),c=colors[it%4])
        plt.savefig(name)

    def correctbBox(self,bBoxes):
        import pdb;pdb.set_trace()
        tempLoc = bBoxes != -5
        bBox = bBoxes[tempLoc]
        return bBox.reshape((-1,5))

    def __call__(self,imDict,name):
        self.drawImage(imDict,name)

class DrawbBoxCenterBatch(object):
    def drawImage(self,imDict,grid,name):

        image = np.transpose(imDict["imData"].numpy(),(0,2,3,1))
        bBoxes  = imDict["bBox"].numpy()
        gridX = grid[0]
        gridY = grid[1]

        colors = ["red","blue","green","yellow","black","brown"]
        totalPassed = 0
        for ik,numOfElement in enumerate(imDict["numOfElement"]):
            # pdb.set_trace()
            tempImage = image[ik]
            tempBBox = bBoxes[totalPassed: totalPassed + numOfElement]
            tempGridX = gridX[totalPassed: totalPassed + numOfElement]
            tempGridY = gridY[totalPassed: totalPassed + numOfElement]

            totalPassed +=numOfElement
            plt.figure()
            plt.imshow(tempImage)
            for it,bBox in enumerate(tempBBox):
                xMin = bBox[1]
                xMax = bBox[3]
                yMin = bBox[2]
                yMax = bBox[4]
                cX = tempGridX[it]
                cY = tempGridY[it]
                plt.plot((xMin,xMin),(yMin,yMax),(xMax,xMax),(yMin,yMax),(xMin,xMax),(yMin,yMin),(xMin,xMax),(yMax,yMax),c=colors[it%4])
                plt.plot(cX*32,cY*32,marker = "*",c=colors[it%4])
            plt.savefig(name+str(numOfElement))

    def __call__(self,imDict,center,name):
        self.drawImage(imDict,center,name)




if __name__=="__main__":
    csv = csvRead("train.txt","/home/osmant/VOCdevkit/VOC2012")
    data = csv()

    voc = VOCDataSet("/home/osmant/VOCdevkit/VOC2012",data,sample = 14)
    outData = voc[7]

    outData.keys()

    crop = RandomCrop()
    outData = crop(outData)

    print(outData["bBox"])

    # flip = RandomFlip()
    # outData = flip(outData)

    # scale = ReScaleData((416,416))
    # outData = scale(outData)

    # normalize = NormalizeData()
    # outData = normalize(outData)

    # eliminate = EliminateSmallBoxes(0.0025)
    # outData = eliminate(outData)

    # tensor = ToTensor()
    # tensor(outData)

    # draw = DrawBbox()
    # draw(outData,"1")

    
