from PIL import Image
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import random
import math
import matplotlib.pyplot as plt
import cv2

def loadScene(path):
    im = Image.open(path +'.png')
    if(im.format == "PNG"):
        im = im.convert("RGB")
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((1024,768))]
    )
    #im = (np.array(im)[:,:,0:3] - [100.32082657,  93.18707606,  89.95372354])/([45.4023921,  45.14495935, 48.19333456])
    im = np.transpose(im,(1,0,2))
    RGB = transform(im).float().to('cuda')
    depth = np.load(path + "_depth.npy")
    #depth = (depth - np.min(depth))/(np.max(depth)-np.min(depth))
    depth = transform(np.transpose(depth,(1,0,2))).float().to('cuda')
    return RGB,depth
#bottleneck layer as described in the MobileNetV2 arch
def getDataMean(iters=100,batchSize=25):
    path = "val/indoors"
    #get random images
    imageList = []
    depthList = []
    #load images for training
    for item in range(batchSize):
        print(item)
        scenes = os.listdir(path)
        scenes = random.sample(scenes,1)[0]
        pathed = path + "/" + scenes
        scans = os.listdir(pathed)
        pickedScan = random.sample(scans,1)[0]
        pathed = pathed +"/" +pickedScan[:33]
        pickedImage = os.listdir(pathed)
        pickedImage = random.sample(pickedImage,1)[0]
        pathed = pathed +"/" +pickedImage[:27]
        im = Image.open(pathed+".png")
        if(im.format == "PNG"):
            im = im.convert("RGB")
        im = np.array(im)
        depth = np.load(pathed + "_depth.npy")
        imageList.append(im)
        depthList.append(depth)
    print(np.array(imageList).shape)
    average = np.sum(imageList,axis=0)
    average = np.sum(average,axis=0)
    average = np.sum(average,axis=0)/(batchSize*1024*768)
  
    dev = np.std(np.array(imageList),axis=(0,1,2))
    print(average,dev)
    
            
class bottleneckLayer(nn.Module):
    def __init__(self,din,squeeze,expand,strides=2,residual=True):
        #residual to be added
        super(bottleneckLayer, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(din,expand*din,1,bias=True),
            nn.BatchNorm2d(expand*din),
            nn.ReLU6()
        )
        #depthwise convolution channel
        self.other = nn.Sequential(
            nn.Conv2d(expand*din,expand*din,3,stride=strides,padding=1,groups=expand*din,bias=True),
            nn.BatchNorm2d(expand*din),
            nn.ReLU6(),
            nn.Conv2d(expand*din,squeeze,1,stride=1,bias=True),
            nn.BatchNorm2d(squeeze),
            nn.ReLU6(),
        )
        self.residuals = residual
        self._weight_init()
        return
    def weightFunction(self,layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
            nn.init.normal_(layer.weight.data,0,.01)
            if layer.bias != None:
                nn.init.constant_(layer.bias.data,0)
        return
    def _weight_init(self):
        self.residual.apply(self.weightFunction)
        self.other.apply(self.weightFunction)
        return
    def forward(self,x):
        if self.residuals == True:
          r = self.residual(x)
          m = self.other(r)
          return m + x
        else:
          r = self.residual(x)
          m = self.other(r)
          return m
class bottleneckLayerMod(nn.Module):
    def __init__(self,din,squeeze,expand,strides=2,residual=True):
        #residual to be added
        super(bottleneckLayerMod, self).__init__()
        self.other = nn.Sequential(
            nn.Conv2d(expand*din,expand*din,3,stride=strides,padding=1,groups=expand*din,bias=False),
            nn.BatchNorm2d(expand*din),
            nn.ReLU6(),
            nn.Conv2d(expand*din,squeeze,1,stride=1,bias=False),
            nn.BatchNorm2d(squeeze),
            nn.ReLU6(),
        )
        self.residuals = residual
        self._weight_init()
        return
    def weightFunction(self,layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
            nn.init.normal_(layer.weight.data,0,.01)
            if layer.bias != None:
                nn.init.constant_(layer.bias.data,0)
        return
    def _weight_init(self):
        self.other.apply(self.weightFunction)
        return
    def forward(self,x):
        if self.residuals == True:
          m = self.other(x)
          return m + x
        else:
          m = self.other(x)
          return m
class depthNet(nn.Module):

    def __init__(self):
        super(depthNet, self).__init__()
        #out 512*256*16
        self.encoderP1 = nn.Sequential(
            nn.Conv2d(3,32,1,2,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            bottleneckLayerMod(32,16,1,1,False),
            bottleneckLayer(16,24,6,residual=False),
            bottleneckLayer(24,24,6,1),
            bottleneckLayer(24,32,6,residual=False),
            bottleneckLayer(32,32,6,1),
            bottleneckLayer(32,32,6,1),
            bottleneckLayer(32,64,6,residual=False),
            bottleneckLayer(64,64,6,1),
            bottleneckLayer(64,64,6,1),
            bottleneckLayer(64,64,6,1),
            bottleneckLayer(64,96,6,1,residual=False),
            bottleneckLayer(96,96,6,1),
            bottleneckLayer(96,96,6,1),
            bottleneckLayer(96,160,6,residual=False),
            bottleneckLayer(160,160,6,1),
            bottleneckLayer(160,160,6,1),
            bottleneckLayer(160,320,6,1,residual=False),
            bottleneckLayer(320,320,6,1),
            bottleneckLayer(320,640,6,1,residual=False),
            bottleneckLayer(640,640,6,1),
            bottleneckLayer(640,960,6,1,residual=False),
            nn.Conv2d(960,1280,1,1,bias=True),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.Conv2d(1280,1024,1,1,bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.PixelShuffle(32),
        )
        self.encoderP1 = self.encoderP1.to('cuda')
        self._weight_init()
        return
    def weightFunction(self,layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
            nn.init.normal_(layer.weight.data,0,.01)
            if layer.bias != None:
                nn.init.constant_(layer.bias.data,0)
        return
    def _weight_init(self):
        self.encoderP1.apply(self.weightFunction)
        return
    def forward(self,x):
        output = self.encoderP1(x)*15
        # newOutput = torch.zeros((5,1024,512)).to('cuda')
        # index = 0
        # for picture in output:
        #   for x in range(newOutput.shape[1]):
        #     for y in range(newOutput.shape[2]):
        #         newOutput[index][x][y] = picture[x%32 + y%32][x//32][y//32].clone()
        #   index += 1
        return output
        #return output.reshape(1,1024,512)
    def getLoss(self,data,goal):
        depth = self.forward(data)
        print(goal.shape,depth.shape)
        sums = torch.sum(torch.square(goal-depth))
        return sums/(depth.shape[2] * depth.shape[3])/26,depth.detach().cpu().numpy()
    def train(self, iters=100000,batchSize=26,lr=0.0003):
        self.optim = optim.Adam(self.encoderP1.parameters(),lr=lr)
        path = "val/indoors"
        #get random images
        for iter in range(iters):
            imageList = []
            depthList = []

            #load images for training
            for item in range(batchSize):
                scenes = os.listdir(path)
                scenes = random.sample(scenes,1)[0]
                pathed = path + "/" + scenes
                scans = os.listdir(pathed)
                pickedScan = random.sample(scans,1)[0]
                pathed = pathed +"/" +pickedScan[:33]
                pickedImage = os.listdir(pathed)
                pickedImage = random.sample(pickedImage,1)[0]
                pathed = pathed +"/" +pickedImage[:27]
                RGB,depth = loadScene(pathed)
                imageList.append(RGB)

                depthList.append(depth)

            #stack used to compile into nice list
            imageList = torch.stack(imageList).to('cuda')
            depthList = torch.stack(depthList).to('cuda')
            #print(image)
            #imageList = torch.permute(imageList,(3,0,1,2))
            self.optim.zero_grad()
            loss,depth =self.getLoss(imageList,depthList)
            loss.backward()
            self.optim.step()
            print(loss,iter)
            if iter % 10 == 0:
                plt.figure()
                plt.imshow(np.array(depth[0].reshape((1024,768))).T,cmap='gray')
                plt.show()
                plt.figure()
                plt.imshow(np.transpose(np.array(depthList[0].detach().cpu()),(1,2,0)).reshape((1024,768)).T,cmap='gray')
                print(np.array(depth[0].reshape((1024,768))).T,np.transpose(np.array(depthList[0].detach().cpu()),(1,2,0)).reshape((1024,768)).T)
                plt.show()
                plt.figure()
                plt.imshow(np.transpose(np.array(imageList[0].detach().cpu()),(2,1,0)))
                plt.show()
if __name__ == "__main__":

    Depth = depthNet().to('cpu')
    state_dict = torch.load('testingModel.pt')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        print(k)
        name = "encoderP1." + k # remove `module.`
        new_state_dict[name] = v
    # load params
    Depth.load_state_dict(new_state_dict)
    Depth.eval()
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    print(frame)
    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        frame = Depth(F.interpolate(torch.permute(torch.tensor(frame).float(),(2,1,0)).view(1,3,640,480),size=(1024,768))).reshape((768,1024)).char().detach().numpy()
        frame = ((frame - np.min(frame))/(np.max(frame) - np.min(frame))*255).astype(np.uint8)
        print(frame)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")
