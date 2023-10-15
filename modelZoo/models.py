from torch import nn
import torchinfo
from torch import nn
import torch



class Res2DModel(nn.Module):
    def __init__(self, outputLen, channels=3, name="Res2DModel", K=1, Dropout=0, HiddenDim=64):
        super().__init__()
        self.name = name
        self.K = K

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
        self.linearStack1 = nn.Sequential(nn.Linear(K*32*8*8, HiddenDim),
                                          nn.LeakyReLU(),
                                          nn.Dropout(Dropout))

        self.classifier = nn.Sequential(nn.Linear(HiddenDim,HiddenDim),
                                        nn.LeakyReLU(),
                                        nn.Dropout(Dropout),
                                        nn.Linear(HiddenDim,outputLen),
                                        nn.Softmax(dim=1))
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
      
        # x = self.gap(x32Out)
         
        x = torch.flatten(x32Out, start_dim=1)
        x = self.linearStack1(x)
        out = self.classifier(x)

        return out
    