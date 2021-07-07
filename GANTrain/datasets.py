import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=(), size=(256,256), unaligned=False, mode='Train'):
        self.transform = transforms.Compose(transforms_)
        if mode=='Train':
            self.labelTransforms = transforms.Compose([transforms_[0],transforms_[1],transforms_[2]])
        else:
            self.labelTransforms = transforms_[0]
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A/%s' % (mode,'img')) + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(root, '%s/A/%s' % (mode,'label')) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B/%s' % (mode,'img')) + '/*.*'))
        self.labels_B = sorted(glob.glob(os.path.join(root, '%s/B/%s' % (mode,'label')) + '/*.*'))
    
        self.toGray = transforms.Grayscale()
        self.resize = transforms.Resize(size)
        self.toTensor = transforms.ToTensor()
#        self.normalize = transforms.Normalize((0.5,),(0.5,))
        
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        label_A = self.labelTransforms(self.toGray(Image.open(self.labels_A[index % len(self.files_A)])))
        label_A = ((np.array(label_A)>0)*1)
        label_A = self.toTensor(label_A)
#        label_A = self.normalize(self.toTensor(label_A))
        label_A = torch.squeeze(label_A,0)
        randIdx = random.randint(0, len(self.files_B) - 1)

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[randIdx]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            
        label_B = self.labelTransforms(self.toGray(Image.open(self.labels_B[randIdx])))
        label_B = ((np.array(label_B)>0)*1)
        label_B = self.toTensor(label_B)
#        label_B = self.normalize(self.toTensor(label_B))
        label_B = torch.squeeze(label_B,0)
        return {'A': item_A, 'B': item_B, 'LA': label_A, 'LB': label_B, 'NA': self.files_A[index % len(self.files_A)], 'NB': self.files_B[index % len(self.files_B)]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))