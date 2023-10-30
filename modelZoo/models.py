from torch import nn
import torchinfo
from torch import nn
import torch
import numpy as np


class Res2DModel(nn.Module):
    def __init__(self, outputLen, inputDim=256, channels=3, name="Res2DModel", K=1, Dropout=0, HiddenDim=64, useGAP=False, useSoftmax=False):
        super().__init__()
        self.name = name
        self.K = K
        self.useGAP = useGAP
        if useGAP:
            self.lin1Dim = K*32
        else:
            downsampledDim = int(np.round(inputDim/2**5))
            self.lin1Dim = K*32*downsampledDim*downsampledDim

        self.convStackK2 = nn.Sequential(nn.BatchNorm2d(channels),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(channels, K*2, (3,3), padding=(1,1))
                                        )
        self.convStackK2_1 = nn.Sequential(nn.BatchNorm2d(K*2),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*2, K*2, (3,3), padding=(1,1)))
        self.convStackK2_2 = nn.Sequential(nn.BatchNorm2d(K*2),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*2, K*4, (3,3), stride=(2,2), padding=(1,1)))

        self.convStackK4 = nn.Sequential(nn.BatchNorm2d(K*4),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*4, K*4, (3,3), padding=(1,1)))
        self.convStackK4_1 = nn.Sequential(nn.BatchNorm2d(K*4),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*4, K*4, (3,3), padding=(1,1)))
        self.convStackK4_2 = nn.Sequential(nn.BatchNorm2d(K*4),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*4, K*8, (3,3), stride=(2,2), padding=(1,1)))

        self.convStackK8 = nn.Sequential(nn.BatchNorm2d(K*8),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*8, K*8, (3,3), padding=(1,1)))
        self.convStackK8_1 = nn.Sequential(nn.BatchNorm2d(K*8),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*8, K*8, (3,3), padding=(1,1)))
        self.convStackK8_2 = nn.Sequential(nn.BatchNorm2d(K*8),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*8, K*16, (3,3), stride=(2,2), padding=(1,1)))

        self.convStackK16 = nn.Sequential(nn.BatchNorm2d(K*16),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*16, K*16, (3,3), padding=(1,1)))
        self.convStackK16_1 = nn.Sequential(nn.BatchNorm2d(K*16),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*16, K*16, (3,3), padding=(1,1)))
        self.convStackK16_2 = nn.Sequential(nn.BatchNorm2d(K*16),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*16, K*32, (3,3), stride=(2,2), padding=(1,1)))
        
        self.convStackK32 = nn.Sequential(nn.BatchNorm2d(K*32),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*32, K*32, (3,3), padding=(1,1)))
        self.convStackK32_1 = nn.Sequential(nn.BatchNorm2d(K*32),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*32, K*32, (3,3), padding=(1,1)))
        self.convStackK32_2 = nn.Sequential(nn.BatchNorm2d(K*32),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*32, K*32, (3,3), stride=(2,2), padding=(1,1)))
        self.convStackK32_3 = nn.Sequential(nn.BatchNorm2d(K*32),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(K*32, K*32, (3,3), padding=(1,1)))

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        
        #nn.Linear(K*16*8*8*depth, HIDDEN_DIM)
        self.linearStack1 = nn.Sequential(nn.Linear(self.lin1Dim, HiddenDim),
                                          nn.LeakyReLU(),
                                          nn.Dropout(Dropout))

        if useSoftmax:
            self.classifier = nn.Sequential(nn.Linear(HiddenDim,HiddenDim),
                                            nn.LeakyReLU(),
                                            nn.Dropout(Dropout),
                                            nn.Linear(HiddenDim,outputLen),
                                            nn.Softmax(dim=1))
        else:
            self.classifier = nn.Sequential(nn.Linear(HiddenDim,HiddenDim),
                                            nn.LeakyReLU(),
                                            nn.Dropout(Dropout),
                                            nn.Linear(HiddenDim,outputLen))
    def forward(self, x):
        x2 = self.convStackK2(x)
        x2_2 = self.convStackK2_1(x2)
        concat2 = torch.add(x2,x2_2)
        x2Out = self.convStackK2_2(concat2)

        x4 = self.convStackK4(x2Out)
        x4 = self.convStackK4_1(x4)
        concat4 = torch.add(x4,x2Out)
        x4Out = self.convStackK4_2(concat4)

        x8 = self.convStackK8(x4Out)
        x8 = self.convStackK8_1(x8)
        concat8 = torch.add(x8,x4Out)
        x8Out = self.convStackK8_2(concat8)

        x16 = self.convStackK16(x8Out)
        x16 = self.convStackK16_1(x16)
        concat16 = torch.add(x16,x8Out)
        x16Out = self.convStackK16_2(concat16)

        x32 = self.convStackK32(x16Out)
        x32 = self.convStackK32_1(x32)
        concat32 = torch.add(x32,x16Out)
        x32 = self.convStackK32_2(concat32)
        x32Out = self.convStackK32_3(x32)
      
        if self.useGAP:
            x = self.gap(x32Out)
            x = torch.flatten(x, start_dim=1)
        else:
            x = torch.flatten(x32Out, start_dim=1)
         
        x = self.linearStack1(x)
        out = self.classifier(x)

        return out
    



class convBlockResDown(torch.nn.Module):
    def __init__(self, chIn, chOut, kernelSize=3, dropout=0.0):
        super().__init__()
        self.padding = int(np.floor(kernelSize/2))
        
        self.act = torch.nn.LeakyReLU()

        self.dropout = torch.nn.Dropout2d(dropout)

        self.bn1 = torch.nn.BatchNorm2d(chIn)
        self.bnRes = torch.nn.BatchNorm2d(chIn)
        self.conv1 = torch.nn.Conv2d(chIn, chIn, kernelSize, padding="same")
        self.bn2 = torch.nn.BatchNorm2d(chIn)
        self.conv2 = torch.nn.Conv2d(chIn, chOut, kernelSize, stride=(2,2), padding=self.padding)
        self.convRes = torch.nn.Conv2d(chIn, chOut, kernelSize, stride=(2,2), padding=self.padding)
        self.convKernel1 = torch.nn.Conv2d(chOut, chOut, 1, padding="same")
        self.bn3 = torch.nn.BatchNorm2d(chOut)
        self.conv3 = torch.nn.Conv2d(chOut, chOut, kernelSize, padding="same")
        self.bn4 = torch.nn.BatchNorm2d(chOut)
        self.conv4 = torch.nn.Conv2d(chOut, chOut, kernelSize, padding="same")
    
    def forward(self, xIn):
        x = self.bn1(xIn)
        x = self.act(x)
        x = self.conv1(x)
        x = self.dropout(x)

        xRes = self.bnRes(xIn)
        xRes = self.act(xRes)
        xRes = self.convRes(xRes)
        xRes = self.convKernel1(xRes)

        x = self.bn2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.dropout(x)
        return x+xRes
    


class ResNet2DModel(nn.Module):
    def __init__(self, outputLen, inputDim=256, channels=3, name="Res2DModel", K=1, Dropout=0.0, HiddenDim=64, useGAP=False, useSoftmax=False):
        super().__init__()
        self.name = name
        self.K = K
        self.useGAP = useGAP
        if useGAP:
            self.lin1Dim = K*32
        else:
            downsampledDim = int(np.round(inputDim/2**5))
            self.lin1Dim = K*32*downsampledDim*downsampledDim

        self.down2 = convBlockResDown(channels,K*2, dropout=Dropout)
        self.down4 = convBlockResDown(K*2,K*4, dropout=Dropout)
        self.down8 = convBlockResDown(K*4,K*8, dropout=Dropout)
        self.down16 = convBlockResDown(K*8,K*16, dropout=Dropout)
        self.down32 = convBlockResDown(K*16,K*32, dropout=Dropout)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        
        #nn.Linear(K*16*8*8*depth, HIDDEN_DIM)
        self.linearStack1 = nn.Sequential(nn.Linear(self.lin1Dim, HiddenDim),
                                          nn.LeakyReLU(),
                                          nn.Dropout(Dropout))

        if useSoftmax:
            self.classifier = nn.Sequential(nn.Linear(HiddenDim,HiddenDim),
                                            nn.LeakyReLU(),
                                            nn.Dropout(Dropout),
                                            nn.Linear(HiddenDim,outputLen),
                                            nn.Softmax(dim=1))
        else:
            self.classifier = nn.Sequential(nn.Linear(HiddenDim,HiddenDim),
                                            nn.LeakyReLU(),
                                            nn.Dropout(Dropout),
                                            nn.Linear(HiddenDim,outputLen))
    def forward(self, x):
        xDown2 = self.down2(x)
        xDown4 = self.down4(xDown2)
        xDown8 = self.down8(xDown4)
        xDown16 = self.down16(xDown8)
        xDown32 = self.down32(xDown16)
      
        if self.useGAP:
            x = self.gap(xDown32)
            x = torch.flatten(x, start_dim=1)
        else:
            x = torch.flatten(xDown32, start_dim=1)
         
        x = self.linearStack1(x)
        out = self.classifier(x)

        return out