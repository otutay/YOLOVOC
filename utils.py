import torch
import numpy as np
import pdb

def loadCheckpoint(fileName):
    data = torch.load(fileName)
    epoch = data['epoch']
    net = data['net']
    optim = data['optim']
    coordLoss = data['coordLoss']
    confLoss = data['confLoss']
    classLoss = data['classLoss']
    lossAll = data['lossAll']
  
    return epoch,net,optim,coordLoss,confLoss,classLoss,lossAll

def saveCheckPoint(fileName,epoch,net,optimizer,coordLoss,confLoss,classLoss,lossAll,isBest):
    data={}
    data['epoch'] = epoch
    data['net'] = net
    data['optim'] = optimizer
    data['coordLoss'] = coordLoss
    data['confLoss'] = confLoss
    data['classLoss'] = classLoss
    data['lossAll'] = lossAll
    if isBest:
        print('data is written to disk as best')
        torch.save(data,'BestDetectionData.pth')
        torch.save(data,fileName)
    else:
        print('data is written to disk')
        torch.save(data,fileName)


def reduceDimen(*args):
    tempArg = tuple()
    for arg in args:
        tempArg += tuple(arg.detach().reshape((-1,1)))
    return tempArg


def loadClassificationParam(fileName,detectNet):
    data = torch.load(fileName)
    classNet = data['net']
    with torch.no_grad():
        for net1,net2 in zip(classNet.named_parameters(),detectNet.named_parameters()):
            if net2[1].shape == net1[1].shape:
                net2[1].data = net1[1].data
    return detectNet           

def cycleParam(startVals,stopVals,percents,totalNum):
    param = []
    for startVal,stopVal,percent in zip(startVals,stopVals,percents):
        tempNum = int((totalNum*percent/100))
        param[-1:] = torch.linspace(startVal,stopVal,tempNum)
    params = torch.tensor(param)
    return params.reshape((1,-1))    


def calcOutput(output):
    classes = np.array(['sheep', 'horse', 'bicycle', 'bottle', 'cow', 'sofa', 'car', 'dog', 'cat', 'person', 'train', 'diningtable', 'aeroplane', 'bus', 'pottedplant', 'tvmonitor', 'chair', 'bird', 'boat', 'motorbike'])           
    anchors = torch.tensor([[1.3221 , 3.19275, 5.05587, 9.47112,11.2364],[1.73145, 4.00944, 8.09892, 4.94053,10.0071]])
    reduction = 32

    predClass = output[:,:,:,:,5:]
    predConf =  torch.sigmoid(output[:,:,:,:,4])
    aboveThr = calcAboveThreshold(predClass,predConf)

    predLoc = torch.zeros_like(output[:,:,:,:,:4])
    predLoc[:,:,:,:,0:2] = output[:,:,:,:,0:2]
    predLoc[:,:,:,:,2:4] = output[:,:,:,:,2:4]

    bBoxes = calcBBox(predLoc,anchors,reduction)
        
    return bBoxes[aboveThr,:]

def calcBBox(predLoc,anchors,reduction):
    
    bSize,wSize,hSize,anchorSize,_ = predLoc.shape
    tempPredLoc = torch.zeros_like(predLoc)
    tempPredLoc[:,:,:,:,0:2] = torch.sigmoid(predLoc[:,:,:,:,0:2])
    tempPredLoc[:,:,:,:,2:4] = torch.exp(predLoc[:,:,:,:,2:4])
        
        # gridX = torch.arange(wSize,requires_grad = False,dtype = torch.float).repeat(hSize,1)
    gridX = torch.arange(wSize,requires_grad = False,dtype = torch.float).reshape((wSize,1)).repeat(1,hSize)
    gridY = torch.arange(hSize,requires_grad = False,dtype = torch.float).reshape(1,hSize).repeat(hSize,1)

    tempPredLoc[:,:,:,:,0] = tempPredLoc[:,:,:,:,0] + gridX.reshape(1,hSize,wSize,1)
    tempPredLoc[:,:,:,:,1] = tempPredLoc[:,:,:,:,1] + gridY.reshape(1,hSize,wSize,1)
    tempPredLoc[:,:,:,:,2] = tempPredLoc[:,:,:,:,2] * anchors[0]
    tempPredLoc[:,:,:,:,3] = tempPredLoc[:,:,:,:,3] * anchors[1]

    bBoxes = convertBBox(tempPredLoc*reduction,reduction)

    return bBoxes

def convertBBox(predLoc,reduction):
    tempBbox = torch.zeros_like(predLoc)
    tempBbox[:,:,:,:,0] = predLoc[:,:,:,:,0] - predLoc[:,:,:,:,2] / 2
    tempBbox[:,:,:,:,1] = predLoc[:,:,:,:,1] - predLoc[:,:,:,:,3] / 2
    tempBbox[:,:,:,:,2] = predLoc[:,:,:,:,0] + predLoc[:,:,:,:,2] / 2
    tempBbox[:,:,:,:,3] = predLoc[:,:,:,:,1] + predLoc[:,:,:,:,3] / 2

    return tempBbox

def calcAboveThreshold(predClass,predConf):
    probOfEachBBox = torch.nn.functional.softmax(predClass,dim = 4)
    probClass,probClassId = torch.max(probOfEachBBox,dim = 4)

    detection = probClass.mul(predConf)
    
    thresholdLoc = detection > 0.02

    return thresholdLoc