import torch
import torch.nn as nn
import torch.optim as optim

from yoloVOC import classTinyArch as architecture
from LossFun import YOLOLoss
from Save import Save
from Data import *
import argparse
import time
from utils import *
import pdb


parser = argparse.ArgumentParser(description="YOLO detection implementation")
parser.add_argument("-b", "--batchSize", default=60,type=int, help="minibatch size, default = 40")
parser.add_argument("-lrMin", "--lrMin", default=5e-5,type=float, help="lr Min, default = 5e-4")
parser.add_argument("-lrMax", "--lrMax", default=1e-3,type=float, help="lr Max, default = 5e-1")
parser.add_argument("-e", "--stopEpoch", default=160,type=int, help="stop epoch, default = 20")
parser.add_argument("-d", "--dir", default="/home/osmant/VOCdevkit/")
parser.add_argument("-dev", "--device", default="cuda")
parser.add_argument("-res", "--resume", default=False,help="resume from a checkpoint,default = False")
parser.add_argument("-dLoad", "--detectLoad",default='detectBest.pth', help="detection model save")
parser.add_argument("-cLoad", "--classLoad",default='BestData.pth', help="data from classification")
parser.add_argument("-log", "--fileName", default='tiny', help="log data")
parser.add_argument("-classNum", "--numOfClass", default=20,type=int, help='class number in dataset,default = 20')
parser.add_argument("-anchorNum", "--anchorNum", default=5,type=int, help='anchor num for each grid, default = 5')
parser.add_argument("-gridNum", "--gridNum", default=13,type=int, help="grid in x,y dimension,default = 13")
parser.add_argument("-imSize", "--imSize", default=416,type=int, help="imSize in x,y dimension,default = 416")
parser.add_argument("-hdf5Save", "--hdf5Save",default="yoloVoc", help="save network results on hdf5")
parser.add_argument("-testEvery", "--testEvery", default=200,type=int, help="test in every [paramNum] iteration")
parser.add_argument("-testNum", "--testNum", default=20,type=int, help="testNumber in every test iteration")
parser.add_argument("-runName", "--runName",default="2", help="runName for logging.")

args = parser.parse_args()
args.startEpoch = 0


write2File = "runName {}, batchSize {}, epoch {}, lrMin {}, lrMax {} \n".format(
    args.runName, args.batchSize, args.stopEpoch, args.lrMin, args.lrMax)
print(write2File)

if args.resume:
    fid = open(args.fileName+'_'+args.runName+'.log', 'a')
    args.startEpoch, net, optim, bestCoordLoss, bestConfLoss, bestClassLoss, bestLossAll = loadCheckpoint(
        args.detectLoad)
else:
    fid = open(args.fileName+'_'+args.runName+'.log', 'w')
    net = architecture()
    optimizer = optim.SGD(net.parameters(), lr=args.lrMin,
                          momentum=0.9, weight_decay=0.005)
    net = loadClassificationParam(args.classLoad, net)
    bestCoordLoss = torch.tensor([1000]).type(torch.float)
    bestConfLoss = torch.tensor([1000]).type(torch.float)
    bestClassLoss = torch.tensor([1000]).type(torch.float)
    bestLossAll = torch.tensor([1000]).type(torch.float)
fid.write(write2File)

save = Save(args.hdf5Save+'_'+args.runName)
save.createFile()
criterion = YOLOLoss(args.numOfClass, device=args.device)
net = net.cpu()

if args.device == "cuda":
    criterion = criterion.to(args.device)
    net = net.to(args.device)
    # net = torch.nn.DataParallel(net).cuda()

trainLoader = getData(args.imSize, file2Read="train.txt",
                      batchSize=args.batchSize)
testLoader = getData(args.imSize, file2Read="val.txt",
                     batchSize=args.batchSize)

testIterator = iter(testLoader)  # manually iterate over test dataset.

numOfIt = int((args.stopEpoch)*len(trainLoader.dataset)/args.batchSize)
print(numOfIt)


lrCycle = cycleParam([1e-5, 1e-4,1e-5,1e-6], [1e-5, 1e-4,1e-5,1e-6], [3, 47,19,31], numOfIt+args.stopEpoch+3)
MCycle = cycleParam([0.90, 0.9], [0.9, 0.9], [50, 50], numOfIt+args.stopEpoch+3)

save.createDataset(lr=lrCycle, momentum=MCycle)
allIt = (args.stopEpoch-args.startEpoch)*len(trainLoader)

# trainIter = iter(trainLoader)
# trainData = next(trainIter)
for it in range(args.startEpoch, args.stopEpoch):
    batchStart = time.time()
    for i, data in enumerate(trainLoader):
    # for i in range(10000):
        # data = trainData
        # print("new-------------------------")
        # print("train")
        net.train()
        tempIt = int(it*len(trainLoader.dataset)/args.batchSize) + i
        optimizer.param_groups[0]['lr'] = lrCycle[0, tempIt].to(args.device)
        optimizer.param_groups[0]['momentum'] = MCycle[0, tempIt].to(args.device)

        inData = data['imData'].cuda(non_blocking=True)
        tempLoc = data["bBox"] != -5
        data["bBox"] = data["bBox"][tempLoc].reshape((-1,5))
        target = (data["bBox"].cuda(non_blocking=True),data["numOfElement"].cuda(non_blocking=True))
        output = net(inData).reshape((inData.shape[0], args.gridNum, args.gridNum, args.anchorNum, -1)) 
        # print("output " , output.max())
        coordLoss, confLoss, classLoss, lossAll = criterion(output, target)
        optimizer.zero_grad()
        lossAll.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
        optimizer.step()
        # dataStart = time.time()
        if it == 0 and i == 0:
            lossAll, coordLoss, confLoss, classLoss = reduceDimen(
                lossAll, coordLoss, confLoss, classLoss)
            save.createDataset(lossAll=lossAll, coordLoss=coordLoss,
                               confLoss=confLoss, classLoss=classLoss)
            # import pdb;pdb.set_trace()
            write2File = "epoch {}/{} ,iteration {}/{}, lr {:.6f}, lossAll {:.3f}/{:.3f}, classLoss{:.3f}/{:.3f}, coordLoss {:.3f}/ {:.3f}, confLoss {:.3f}/{:.3f}, time{:.3f} \n".format(
                it, args.stopEpoch, tempIt, allIt, lrCycle[0, tempIt], lossAll.item(), bestLossAll.item(), classLoss.item(), bestClassLoss.item(), coordLoss.item(), bestCoordLoss.item(), confLoss.item(), bestConfLoss.item(), time.time() - batchStart)
            print(write2File)
            fid.write(write2File)
            fid.flush()
            batchStart = time.time()
        elif i % 1 == 0:
            lossAll, coordLoss, confLoss, classLoss = reduceDimen(
                lossAll, coordLoss, confLoss, classLoss)
            save.variable2Write(
                lossAll=lossAll, coordLoss=coordLoss, confLoss=confLoss, classLoss=classLoss)
            write2File = "epoch {}/{} ,iteration {}/{}, lr {:.6f}, lossAll {:.3f}/{:.3f}, classLoss{:.3f}/{:.3f}, coordLoss {:.3f}/ {:.3f}, confLoss {:.3f}/{:.3f}, time{:.3f}  \n".format(
                it, args.stopEpoch, tempIt, allIt, lrCycle[0, tempIt], lossAll.item(), bestLossAll.item(), classLoss.item(), bestClassLoss.item(), coordLoss.item(), bestCoordLoss.item(), confLoss.item(), bestConfLoss.item(), time.time() - batchStart)
            print(write2File)
            fid.write(write2File)
            fid.flush()
            batchStart = time.time()
            # dataStart = time.time()
        
        if i % args.testEvery == 0:
            net.eval()
            classLossSum = 0
            confLossSum = 0
            coordLossSum = 0
            lossAllSum = 0
            for k in range(args.testNum):
                try:
                    testData = next(testIterator)
                except:
                    testIterator = iter(testLoader)
                    testData = next(testIterator)

                with torch.no_grad():
                    inData = testData['imData'].type(torch.float).cuda(non_blocking=True)
                    tempLoc = testData["bBox"] != -5
                    testData["bBox"] = testData["bBox"][tempLoc].reshape((-1,5))
                    target = (testData['bBox'].cuda(non_blocking=True), testData["numOfElement"].cuda(non_blocking=True))
                    output = net(inData).reshape((inData.shape[0], args.gridNum, args.gridNum, args.anchorNum, -1))
                    coordLoss, confLoss, classLoss, lossAll = criterion(
                        output, target)
                    coordLossSum += coordLoss.item()
                    confLossSum += confLoss.item()
                    classLossSum += classLoss.item()
                    lossAllSum += lossAll.item()

            lossAllAvg = lossAllSum/args.testNum
            coordLossAvg = coordLossSum/args.testNum
            confLossAvg = confLossSum/args.testNum
            classLossAvg = classLossSum/args.testNum

            write2File = "Test In -> epoch {}/{}, valLossAll {:.4f}, valCoordLoss {:.4f}, valConfLoss {:.4f}, valClassLoss {:.4f}  \n".format(
                it, args.stopEpoch, lossAllAvg, coordLossAvg, confLossAvg, classLossAvg)
            print(write2File)
            fid.write(write2File)
            fid.flush()
            if it == 0 and i == 0:
                lossAllAvg, coordLossAvg, confLossAvg, classLossAvg = reduceDimen(torch.tensor(
                    lossAllAvg), torch.tensor(coordLossAvg), torch.tensor(confLossAvg), torch.tensor(classLossAvg))
                save.createDataset(valAllLossAvg=lossAllAvg, vallCoordLossAvg=coordLossAvg,
                                   ValConfLossAvg=confLossAvg, ValClassLossAvg=classLossAvg)
            else:
                lossAllAvg, coordLossAvg, confLossAvg, classLossAvg = reduceDimen(torch.tensor(
                    lossAllAvg), torch.tensor(coordLossAvg), torch.tensor(confLossAvg), torch.tensor(classLossAvg))
                save.variable2Write(valAllLossAvg=lossAllAvg, vallCoordLossAvg=coordLossAvg,
                                    ValConfLossAvg=confLossAvg, ValClassLossAvg=classLossAvg)

            if lossAllAvg < bestLossAll:
                bestLossAll = lossAllAvg
                bestCoordLoss = coordLossAvg
                bestConfLoss = confLossAvg
                bestClassLoss = classLossAvg
                saveCheckPoint('TinyDetectionCheckpoint.pth', it, net, optimizer,
                               bestCoordLoss, bestConfLoss, bestClassLoss, bestLossAll, True)
            else:
                saveCheckPoint('TinyDetectionCheckpoint.pth', it, net, optimizer,
                               bestLossAll, bestCoordLoss, bestConfLoss, bestClassLoss, False)

            batchStart = time.time()
