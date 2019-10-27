import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
class classTinyArch(nn.Module):
    def __init__(self):
        super(classTinyArch,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,stride=1,padding=1)
        self.conv1B = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16,32,3,stride = 1,padding = 1)
        self.conv2B = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32,64,3,stride = 1,padding = 1)
        self.conv3B = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64,128,3,stride = 1,padding = 1)
        self.conv4B = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128,256,3,stride = 1,padding = 1)
        self.conv5B = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256,512,3,stride = 1,padding = 1)
        self.conv6B = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512,1024,3,stride = 1,padding = 1)
        self.conv7B = nn.BatchNorm2d(1024)
        
        self.conv8 = nn.Conv2d(1024,1024,3,stride = 1, padding = 1)
        self.conv8B = nn.BatchNorm2d(1024)

        self.conv9 = nn.Conv2d(1024,125,1,stride=1,padding=0)
        # self.conv8 = nn.Conv2d(1024,1000,1)

    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv1B(x)

        x = F.max_pool2d(x,2,2)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv2B(x)
        x = F.max_pool2d(x,2,2)

        x = self.conv3(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv3B(x)
        x = F.max_pool2d(x,2,2)

        x = self.conv4(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv4B(x)
        x = F.max_pool2d(x,2,2)

        x = self.conv5(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv5B(x)
        x = F.max_pool2d(x,2,2)

        x = self.conv6(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv6B(x)
        # x = F.max_pool2d(x,2,1)


        x = self.conv7(x)
        x = F.leaky_relu(x,0.1)
        x = self.conv7B(x)


        x = self.conv8(x)
        x = F.leaky_relu(x)
        x = self.conv8B(x)

        x = self.conv9(x)
        x = x.permute(0,2,3,1)
        return x



if __name__=="__main__":
    c = classTinyArch()
    # pdb.set_trace()
    x = torch.ones((1,3,448,448))
    c(x)
