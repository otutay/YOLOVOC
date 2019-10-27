import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import pdb

class Show():
    def __init__(self,fileName):
        self.fileName = fileName
        self.fid = 0
        self.limit = 5

    def openFile(self):
        # print(self.fileName)
        self.fid = h5py.File(self.fileName,'r')

    def closeFile(self):
        self.fid.close()

    def getHier(self):
        return list(self.fid.keys())

    def extractData(self,param):
        data = np.array(self.fid[param])
        return data

    def plotNormal(self):
        self.openFile()
        hier = self.getHier()
        hier2Plot = self.dataNotContain(hier,'.')
       
        valName = self.dataContain(hier2Plot,'val')
        trainName = self.dataNotContain(hier2Plot,'val')
        self.plotData(trainName,figureName ="train") 
        self.plotData(valName,figureName ="val") 

        self.closeFile()
        
    def plotData(self,name,figureName = "train"):
        # pdb.set_trace()
        lr = self.extractData('lr')
        tempName = self.dataNotContain(name,'lr')
        tempName = self.dataNotContain(tempName,'momentum')
        size = len(tempName)
        # pdb.set_trace()
        
        plt.figure()
        for it,name in enumerate(tempName):
            data = self.extractData(name)
            data = self.clamp(data,self.limit)
            tempLr = lr[0,:data.shape[0]]
            
            plt.subplot(2,1,1)
            plt.plot(data,label=name)
            plt.legend(loc='best')
            plt.subplot(2,1,2)
            plt.plot(lr.reshape((-1)),label=name)
            # plt.show()
        plt.savefig(figureName+'.png',dpi = 500)

    def clamp(self,data,limit):
        loc = data >limit
        data[loc] = limit
        return data
    def dataContain(self,data,data2Find):
        tempOut = [ x  for x in data if x.find(data2Find)>-1]
        return tempOut

    def dataNotContain(self,data,data2Find):
        tempOut = [ x for x in data if x.find(data2Find)==-1]
        return tempOut


if __name__=="__main__":
    show = Show("yoloVoc_2.hdf5")
    show.plotNormal()
    # show.plotImageData()
