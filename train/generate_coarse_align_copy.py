import sys 
import PIL.Image as Image 
import os 
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import warnings
import torch.nn.functional as F
import glob
sys.path.append('/home2/shaon/RANSAC-Flow/utils/')
import outil
import torchvision.models as models
import re
import cv2

if not sys.warnoptions:
    warnings.simplefilter("ignore")

sys.path.append('/home2/shaon/RANSAC-Flow/model')
from resnet50 import resnet50

import torchvision.models as models
import kornia.geometry as tgm
import matplotlib.pyplot as plt 
# %matplotlib inline 
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

files_path = glob.glob("/ssd_scratch/cvit/shaon/forms_org/*.jpg")
res = sorted(files_path, key = lambda x: x.split("/")[-1].split("_")[0])
if os.path.exists("/ssd_scratch/cvit/shaon/forms_org/1_2.jpg"):
	print("present")

# filenames = os.listdir("/ssd_scratch/cvit/shaon/forms_org/*.jpg")
# print(filenames)
length = len(files_path)
# res.sort(key = num_sort)
# print(res[4:7])
# print(files_path.sort())
# print(files_path[0:3])

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

j = 0
count = 0
triplet_path = []
while j <= length:
	temp = "/ssd_scratch/cvit/shaon/forms_org/" + str(count) + "_" + "1" + ".jpg"
	im1 = "/ssd_scratch/cvit/shaon/forms_org/" + str(count) + "_" + "2" + ".jpg"
	im2 = "/ssd_scratch/cvit/shaon/forms_org/" + str(count) + "_" + "3" + ".jpg"
	triplet_path.append(temp)
	triplet_path.append(im1)
	triplet_path.append(im2)
	print(triplet_path)



	# triplet_path = files_path[j : j + 3]
	for img_path in range(1,3):
		
		print(img_path)
		img1 = triplet_path[0]
		img2 = triplet_path[img_path]
		print(triplet_path[img_path])
		# I1 = Image.open(img1).convert('RGB')
		# I2 = Image.open(img2).convert('RGB')
		I1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
		print(I1.shape)
		I2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)
		print(I2.shape)
		I1 = Image.fromarray(I1)
		I2 = Image.fromarray(I2)



	#Pre-processing images (multi-scale + imagenet normalization)
	## We only compute 3 scales : 
		I1Down2 = outil.resizeImg(I1, strideNet, minSize // 2)
		I1Up2 = outil.resizeImg(I1, strideNet, minSize * 2)
		I1 = outil.resizeImg(I1, strideNet, minSize)
		I1Tensor = transforms.ToTensor()(I1).unsqueeze(0).cuda()
		print(I1Down2.size)
		print(I1Up2.size)


		feat1Down2 = F.normalize(resNetfeat(preproc(I1Down2).unsqueeze(0).cuda()))
		feat1 = F.normalize(resNetfeat(preproc(I1).unsqueeze(0).cuda()))
		feat1Up2 = F.normalize(resNetfeat(preproc(I1Up2).unsqueeze(0).cuda()))


		I2 = outil.resizeImg(I2, strideNet, minSize)
		I2Tensor = transforms.ToTensor()(I2).unsqueeze(0).cuda()
		feat2 = F.normalize(resNetfeat(preproc(I2).unsqueeze(0).cuda()))
		print(I2.size)

#Extract matches
		W1Down2, H1Down2 = outil.getWHTensor(feat1Down2)
		W1, H1 = outil.getWHTensor(feat1)
		W1Up2, H1Up2 = outil.getWHTensor(feat1Up2)
		print(W1, H1)


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
			print("not>>> good match")
			print ('not a good pair...')    
		bestParam, bestInlier, match1Inlier, match2Inlier = outil.RANSAC(nbIter, match1, match2, tolerance, nbPoint, Transform)


## We keep the pair only we have enough inliers
		print(len(match1Inlier))
		if len(match1Inlier) > 50 : 
			if transform == 'Affine':
				grid = F.affine_grid(torch.from_numpy(bestParam[:2].astype(np.float32)).unsqueeze(0).cuda(), IpTensor.size()) # theta should be of size N×2×3
			else : 
				warper = tgm.HomographyWarper(I1Tensor.size()[2],  I1Tensor.size()[3])
				grid = warper.warp_grid(torch.from_numpy(bestParam.astype(np.float32)).unsqueeze(0).cuda())
			I2Sample = F.grid_sample(I2Tensor.clone(), grid)

			print("entering...")

    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(I1)
    # plt.savefig("/home2/shaon/form_train/5_1.jpg")
    # plt.close()
    # I1.save("/home2/shaon/form_train/4_1.jpg")


			I2 = transforms.ToPILImage()(I2Sample.squeeze().cpu())
			print(I2.size)
			print(I1.size)
			img2_sp = img2.split("/")
			img1_sp = img1.split("/")
			source_path2  = "/ssd_scratch/cvit/shaon/forms_train" + "/"+ img2_sp[-1]
			I2.save(source_path2)
			source_path1 = "/ssd_scratch/cvit/shaon/forms_train" + "/" + img1_sp[-1]
			I1.save(source_path1)
			print(source_path1, flush = True)
			print(source_path2, flush  =  True)
			

	print(j)
	j+=3
	count +=1
	triplet_path = []




    # plt.subplot(1,2,1)
    # plt.axis("off")
    # plt.imshow(transforms.ToPILImage()(I2Sample.squeeze().cpu()))
    # plt.savefig("/home2/shaon/form_train/5_3.jpg")
    # plt.close()







