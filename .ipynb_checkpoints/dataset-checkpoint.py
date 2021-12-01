#written by mohomin123@gmail.com
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pickle as pkl
import random
import json
import skimage.draw

from PIL import Image, ImageDraw
import os
import numpy as np
import sys

#copied from torchvision
from engine import train_one_epoch, evaluate
import utils
from torchvision import transforms as T

import shutil
import heapq

class ViennaDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None, test_mode = False):
        self.root = root
        self.transforms = transforms
        self.test_mode = test_mode
        # Load images, sorting them
        if test_mode:
            self.imgs = list(sorted(os.listdir(root)))
        else:
            self.imgs = list(sorted(os.listdir(os.path.join(root, "00/image"))))

    def __getitem__(self, idx):

        if self.test_mode:
            img_path = os.path.join(self.root, self.imgs[idx])
            img = Image.open(img_path).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            return img, self.imgs[idx]
            
        img_path = os.path.join(self.root, "00/image", self.imgs[idx])
        mask_path = os.path.join(self.root, "00/mask", self.imgs[idx].replace(".jpg","_mask.png"))
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        np.set_printoptions(threshold=sys.maxsize)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin,ymin,xmax,ymax])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        code = int(self.imgs[idx][0:2])
        labels = torch.full((num_objs,), 0, dtype=torch.int64)
        if code == 1:
            labels = torch.full((num_objs,), 1, dtype=torch.int64)
        elif code == 4:
            labels = torch.full((num_objs,), 1, dtype=torch.int64)
        elif code == 5:
            labels = torch.full((num_objs,), 1, dtype=torch.int64)
        elif code == 7:
            labels = torch.full((num_objs,), 2, dtype=torch.int64)
        elif code == 10:
            labels = torch.full((num_objs,), 2, dtype=torch.int64)
        elif code == 11:
            labels = torch.full((num_objs,), 2, dtype=torch.int64)
        elif code == 12:
            labels = torch.full((num_objs,), 3, dtype=torch.int64)
        elif code == 15:
            labels = torch.full((num_objs,), 2, dtype=torch.int64)
        elif code == 37:
            labels = torch.full((num_objs,), 4, dtype=torch.int64)
        else:
            print("code non exist")
        """
        human = [1,4,5,7,10,11,12,15,37]
        sun = [8]
        planet = [19]
        heart = [35]
        ceramic = [14]
        bottle = [48]
        bowl = [30]
        star = [3]
        jewel = [6]
        moon = [9]
        triangle = [47]
        arrow = [52]
        rect = [43]
        shield = [55]
        building = [13]
        tree = [46]
        flower = [51]
        car = [53]
        airplane = [26]
        crown = [2]
        hat = [34]
        cross = [27, 32]
        landscape = [38]
        mountain = [40]
        plantac = [42]
        leaf = [18]
        fruit = [49]
        tool = [33, 45]
        """
        
        #labels = torch.full((num_objs,), int(self.imgs[idx][0:2]), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels #read folder name
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd #this image has single object?
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
 
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
 
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
 
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)
    