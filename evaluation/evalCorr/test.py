from coarseAlignFeatMatch import CoarseAlign
import sys
sys.path.append('/home2/shaon/RANSAC-Flow/utils')
import outil

 
sys.path.append('/home2/shaon/RANSAC-Flow/model')
import model as model

import PIL.Image as Image 
import os 
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
import warnings
import torch.nn.functional as F
import pickle 
import pandas as pd
import kornia.geometry as tgm
from scipy.misc import imresize
from itertools import product

if not sys.warnoptions:
    warnings.simplefilter("ignore")
# import matplotlib.pyplot as plt 
# %matplotlib inline 

## composite image    
def get_Avg_Image(Is, It) : 
    
    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))


resumePth = '/ssd_scratch/cvit/shaon/pretrained_models/MegaDepth_Theta1_Eta001_Grad1_0.774.pth' ## model for visualization
kernelSize = 7

Transform = outil.Homography
nbPoint = 4
    

## Loading model
# Define Networks
network = {'netFeatCoarse' : model.FeatureExtractor(), 
           'netCorr'       : model.CorrNeigh(kernelSize),
           'netFlowCoarse' : model.NetFlowCoarse(kernelSize), 
           'netMatch'      : model.NetMatchability(kernelSize),
           }
    

for key in list(network.keys()) : 
    network[key].cuda()
    typeData = torch.cuda.FloatTensor

# loading Network 
param = torch.load(resumePth)
msg = 'Loading pretrained model from {}'.format(resumePth)
# print (msg)

for key in list(param.keys()) : 
    network[key].load_state_dict( param[key] ) 
    network[key].eval()


I1 = Image.open('/home2/shaon/RANSAC-Flow/test_images/test_1.jpg').convert('RGB')
I2 = Image.open('/home2/shaon/RANSAC-Flow/test_images/target_1.jpg').convert('RGB')

#7 scales, setting ransac parameters
nbScale = 7
coarseIter = 10000
coarsetolerance = 0.05
minSize = 400
imageNet = True # we can also use MOCO feature here
scaleR = 1.2 

coarseModel = CoarseAlign(nbScale, coarseIter, coarsetolerance, 'Homography', minSize, 1, True, imageNet, scaleR)
# # print("hello")
coarseModel.setSource(I1)
coarseModel.setTarget(I2)

I2w, I2h = coarseModel.It.size
featt = F.normalize(network['netFeatCoarse'](coarseModel.ItTensor))
            
#### -- grid     
gridY = torch.linspace(-1, 1, steps = I2h).view(1, -1, 1, 1).expand(1, I2h,  I2w, 1)
gridX = torch.linspace(-1, 1, steps = I2w).view(1, 1, -1, 1).expand(1, I2h,  I2w, 1)
grid = torch.cat((gridX, gridY), dim=3).cuda() 
warper = tgm.HomographyWarper(I2h,  I2w)

bestPara, InlierMask = coarseModel.getCoarse(np.zeros((I2h, I2w)))
bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()

#Coarse Alignment
flowCoarse = warper.warp_grid(bestPara)
I1_coarse = F.grid_sample(coarseModel.IsTensor, flowCoarse)
I1_coarse_pil = transforms.ToPILImage()(I1_coarse.cpu().squeeze())

#Fine Alignment
featsSample = F.normalize(network['netFeatCoarse'](I1_coarse.cuda()))


corr12 = network['netCorr'](featt, featsSample)
flowDown8 = network['netFlowCoarse'](corr12, False) ## output is with dimension B, 2, W, H

flowUp = F.interpolate(flowDown8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
flowUp = flowUp.permute(0, 2, 3, 1)

flowUp = flowUp + grid

flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()

I1_fine = F.grid_sample(coarseModel.IsTensor, flow12)
I1_fine_pil = transforms.ToPILImage()(I1_fine.cpu().squeeze())

I1_fine_pil.save("/home2/shaon/RANSAC-Flow/results/1.jpg")
I2.save("/home2/shaon/RANSAC-Flow/results/target1.jpg")