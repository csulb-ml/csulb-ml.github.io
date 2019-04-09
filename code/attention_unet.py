import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P

def unet_block(chanIn, chanOut, ks=1, stride=1):
    return nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(chanOut, chanOut, kernel_size=3,stride=stride, padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(inplace=True)
            )

def unet_block_att(chanIn, chanOut, ks=1, stride=1):
    return nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=3,stride=stride,padding=1),      
            nn.BatchNorm2d(chanOut),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(chanOut, chanOut, kernel_size=3,stride=stride, padding=1),     
            nn.BatchNorm2d(chanOut),
            nn.ReLU(inplace=True),
            SelfAttention(chanOut)
            )

#fastai implimentation
class SelfAttention(nn.Module):
    def __init__(self, n_channels:int):
        super().__init__()
        self.query = nn.Conv1d(n_channels, n_channels//8,1)
        self.key   = nn.Conv1d(n_channels, n_channels//8,1)
        self.value = nn.Conv1d(n_channels, n_channels,1)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        #Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax((torch.bmm(f.permute(0,2,1).contiguous(), g)), dim=1)
        o = self.gamma * torch.bmm(h, beta) #+x
        return o.view(*size).contiguous()    

class Attention_Unet(nn.Module):
    def __init__(self, input_size, attention_2:bool=False, attention_bottom:bool=False, attention_up:bool=False):
        super().__init__()
        self.chn = input_size
        self.attention_2 = attention_2
        self.attention_bottom = attention_bottom
        self.attention_up = attention_up       
        self.block1 = unet_block(1, self.chn)
        self.pool1 = nn.MaxPool2d(2)        
        self.block2 = unet_block(self.chn, self.chn*2)
        self.block2_att = unet_block_att(self.chn, self.chn*2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottom_layer = unet_block(self.chn*2, self.chn*4)
        self.bottom_att = unet_block_att(self.chn*2, self.chn*4)
                 
        self.upsamp1 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.up1_1x1 = nn.Sequential( 
                        nn.Conv2d(self.chn*4, self.chn*2, kernel_size=1),
                        nn.ReLU(inplace=True)
                        ) 
        self.upconv1 = unet_block(self.chn*4, self.chn*2)
        self.upconv1_att = unet_block_att(self.chn*4, self.chn*2)
        
        self.upsamp2 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.up2_1x1 = nn.Sequential(
                        nn.Conv2d(self.chn*2, self.chn, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.upconv2 = unet_block(self.chn*2, self.chn)
        self.out = nn.Conv2d(self.chn, 1,kernel_size=1)

    def forward(self, x):
        skip1 = self.block1(x) 
        pool_1 = self.pool1(skip1)

        if self.attention_2:
            skip2 = self.block2_att(pool_1) 
        else:
            skip2 = self.block2(pool_1)  

        pool_2 = self.pool2(skip2)  

        if self.attention_bottom:
            bottom = self.bottom_att(pool_2)
        else:
            bottom = self.bottom_layer(pool_2)

        up1 = self.upsamp1(bottom) 
        up1 = self.up1_1x1(up1)   
        cat1 = torch.cat([skip2,up1],dim=1) 
        
        if self.attention_up:
            out1 = self.upconv1_att(cat1)
        else:
            out1 = self.upconv1(cat1)
        
        up2 = self.upsamp2(out1)
        up2 = self.up2_1x1(up2)   
        cat2 = torch.cat([skip1,up2],dim=1) 
        x = self.upconv2(cat2)        
        x = self.out(x) 
        return x