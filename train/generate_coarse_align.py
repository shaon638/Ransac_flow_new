import sys 
import PIL.Image as Image 
import os 
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import warnings
import torch.nn.functional as F

sys.path.append('/home2/shaon/RANSAC-Flow/utils/')
import outil
import torchvision.models as models


if not sys.warnoptions:
    warnings.simplefilter("ignore")

sys.path.append('/home2/shaon/RANSAC-Flow/model')
from resnet50 import resnet50

import torchvision.models as models
import kornia.geometry as tgm
import matplotlib.pyplot as plt 
# %matplotlib inline 

minSize = 480 # min dimension in the resized image
nbIter = 10000 # nb Iteration
tolerance = 0.05 # tolerance
transform = 'Homography' # coarse transformation
strideNet = 16 # make sure image size is multiple of strideNet size
MocoFeat = True ## using moco feature or not

### ImageNet normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preproc = transforms.Compose([transforms.ToTensor(), normalize,])

#loading model(Moco Feature or ImageNet Feature)
if MocoFeat : 
    resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3']
    resNetfeat = resnet50()
    featPth = '/ssd_scratch/cvit/shaon/pretrained_models/resnet50_moco.pth'
    param = torch.load(featPth)
    state_dict = {k.replace("module.", ""): v for k, v in param['model'].items()}
    msg = 'Loading pretrained model from {}'.format(featPth)
    print (msg)
    resNetfeat.load_state_dict( state_dict )       

else : 
    resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3']
    resNetfeat = models.resnet50(pretrained=True)           
resnet_module_list = [getattr(resNetfeat,l) for l in resnet_feature_layers]
last_layer_idx = resnet_feature_layers.index('layer3')
resNetfeat = torch.nn.Sequential(*resnet_module_list[:last_layer_idx+1])

resNetfeat.cuda()
resNetfeat.eval()

if transform == 'Affine' :
 
    Transform = outil.Affine
    nbPoint = 3
    
else : 
    Transform = outil.Homography
    nbPoint = 4


img1 = '/ssd_scratch/cvit/shaon/form/form_train_org/5_1.jpg'
img2 = '/ssd_scratch/cvit/shaon/form/form_train_org/5_2.jpg'
I1 = Image.open(img1).convert('RGB')
I2 = Image.open(img2).convert('RGB')

#Pre-processing images (multi-scale + imagenet normalization)
## We only compute 3 scales : 
I1Down2 = outil.resizeImg(I1, strideNet, minSize // 2)
I1Up2 = outil.resizeImg(I1, strideNet, minSize * 2)
I1 = outil.resizeImg(I1, strideNet, minSize)
I1Tensor = transforms.ToTensor()(I1).unsqueeze(0).cuda()


feat1Down2 = F.normalize(resNetfeat(preproc(I1Down2).unsqueeze(0).cuda()))
feat1 = F.normalize(resNetfeat(preproc(I1).unsqueeze(0).cuda()))
feat1Up2 = F.normalize(resNetfeat(preproc(I1Up2).unsqueeze(0).cuda()))


I2 = outil.resizeImg(I2, strideNet, minSize)
I2Tensor = transforms.ToTensor()(I2).unsqueeze(0).cuda()
feat2 = F.normalize(resNetfeat(preproc(I2).unsqueeze(0).cuda()))

#Extract matches
W1Down2, H1Down2 = outil.getWHTensor(feat1Down2)
W1, H1 = outil.getWHTensor(feat1)
W1Up2, H1Up2 = outil.getWHTensor(feat1Up2)


featpMultiScale = torch.cat((feat1Down2.contiguous().view(1024, -1), feat1.contiguous().view(1024, -1), feat1Up2.contiguous().view(1024, -1)), dim=1)
WMultiScale = torch.cat((W1Down2, W1, W1Up2))
HMultiScale = torch.cat((H1Down2, H1, H1Up2))

W2, H2 = outil.getWHTensor(feat2)
        
feat2T = feat2.contiguous().view(1024, -1) 
        
        
## get mutual matching
index1, index2 = outil.mutualMatching(featpMultiScale, feat2T)
W1MutualMatch = WMultiScale[index1]
H1MutualMatch = HMultiScale[index1]

W2MutualMatch = W2[index2]
H2MutualMatch = H2[index2]


ones = torch.cuda.FloatTensor(H2MutualMatch.size(0)).fill_(1)
match2 = torch.cat((H1MutualMatch.unsqueeze(1), W1MutualMatch.unsqueeze(1), ones.unsqueeze(1)), dim=1)
match1 = torch.cat((H2MutualMatch.unsqueeze(1), W2MutualMatch.unsqueeze(1), ones.unsqueeze(1)), dim=1)

#RANSAC
## if very few matches, it is probably not a good pair
if len(match1) < nbPoint : 
    print ('not a good pair...')    
bestParam, bestInlier, match1Inlier, match2Inlier = outil.RANSAC(nbIter, match1, match2, tolerance, nbPoint, Transform)


## We keep the pair only we have enough inliers
if len(match1Inlier) > 50 : 
                
    if transform == 'Affine':
        grid = F.affine_grid(torch.from_numpy(bestParam[:2].astype(np.float32)).unsqueeze(0).cuda(), IpTensor.size()) # theta should be of size N×2×3
    else : 

        warper = tgm.HomographyWarper(I1Tensor.size()[2],  I1Tensor.size()[3])
        grid =  warper.warp_grid(torch.from_numpy(bestParam.astype(np.float32)).unsqueeze(0).cuda())


    I2Sample = F.grid_sample(I2Tensor.clone(), grid)

    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(I1)
    # plt.savefig("/home2/shaon/form_train/5_1.jpg")
    # plt.close()
    # I1.save("/home2/shaon/form_train/4_1.jpg")
    I2 = transforms.ToPILImage()(I2Sample.squeeze().cpu())
    I2.save("/home2/shaon/form_train/5_2.jpg")


    # plt.subplot(1,2,1)
    # plt.axis("off")
    # plt.imshow(transforms.ToPILImage()(I2Sample.squeeze().cpu()))
    # plt.savefig("/home2/shaon/form_train/5_3.jpg")
    # plt.close()







