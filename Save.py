import h5py
import torch
import time
import pdb

class Save():
    def __init__(self, name2Save):
        self.name2Save = name2Save + '.hdf5'
        self.fid = 0

    def createFile(self):
        """Create file for the first time"""
        # print("file name 2 write",self.name2Save)
        self.fid = h5py.File(self.name2Save,'w')
        self.fid.close() 

    def openFile(self):
        """open hdf5 for writing"""
        # print("open File 2 write")
        # print(self.name2Save)
        self.fid = h5py.File(self.name2Save,'a')

    def closeFile(self):
        """close already opened file"""
        # print("close file")
        self.fid.close()

    def readHier(self):
        """Read file hierarchy"""
        # myHier = []
        # self.fid.visit(myHier.append)
        # return myHier
        myHier = list(self.fid.keys())
        return myHier

    def createDataset(self, **kwargs):
        self.openFile()
        for key,value in kwargs.items():
            shape = list(value.shape)
            shape[0] = None # make data shape unlimited.
            # pdb.set_trace()
            self.fid.create_dataset(key,value.shape,data = value.cpu(),maxshape=shape)
        self.closeFile()

    def variable2Write(self,**kwargs):
        self.openFile()
        for key,value in kwargs.items():
            tempSize = list(self.fid[key].shape)
            tempSize[0] = tempSize[0] + 1
            self.fid[key].resize((tempSize))
            self.fid[key][-1] = value.cpu()
        self.closeFile()

    def readFile(self):
        self.openFile()
        hier = self.readHier()
        for it in hier:
            print(torch.tensor(self.fid[it]).shape)
            # print(torch.tensor(self.fid[it]))
        self.closeFile()
    
if __name__ == "__main__":
    save = Save("dummy")
    save.createFile()
    a = 10*torch.ones((1,4,2,3,3))
    save.createDataset(weight = a)
    a = 11*torch.ones((1,4,2,3,3))
    save.variable2Write(weight = a)
    a = 12*torch.ones((1,4,2,3,3))
    save.variable2Write(weight = a)
    b = 5*torch.ones((1,4,2,3,3))
    save.createDataset(weight2 = b)
    a = 6*torch.ones((1,4,2,3,3))
    save.variable2Write(weight2 = a)
    a = 7*torch.ones((1,4,2,3,3))
    save.variable2Write(weight2 = a)
    save.readFile()
