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

"""
data에는 __getitem__과 __len__이 있어야 함.
__getitem__은 image, target을 반환해야 함.
__len__은 이미지의 길이를 반환.
"""
class ViennaDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Load images, sorting them
        self.imgs = list(sorted(os.listdir(os.path.join(root, "crown_woman/images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "crown_woman/mask"))))


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "crown_woman/images", self.imgs[idx])
        mask_path = os.path.join(self.root, "crown_woman/mask", self.masks[idx])
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
        labels = torch.full((num_objs,), int(self.imgs[idx][0]), dtype=torch.int64)
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
        #print(img)
        #print(target)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

class PennFudanDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Load images, sorting them
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)

        mask = np.array(mask)
        np.set_printoptions(threshold=sys.maxsize)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]
        #print(masks)
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
        labels = torch.ones((num_objs,),dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #print(img)
        #print(target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def save_mask_from_json(json_path,img_path,save_path):
    fp = open(json_path,'r')
    json_data = json.load(fp)
    fp.close()
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for key in json_data.keys():
        # get information
        file_data = json_data[key]
        img_name = file_data['filename']
        img_size = file_data['size']
        mask_regions = file_data['regions']
        if len(mask_regions) == 0: continue
        region_points = mask_regions[0]['shape_attributes']
        all_points_x = region_points['all_points_x']
        all_points_y = region_points['all_points_y']
        img = Image.open(img_path+img_name)

        # draw mask
        mask = np.zeros((img.size[1],img.size[0]))
        rr, cc = skimage.draw.polygon(all_points_y,all_points_x)
        mask[rr,cc] = 1
        # save mask
        img2 = Image.fromarray(mask)
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        fp2 = open(save_path+img_name.replace('.jpg','_mask.jpg'), 'w')
        img2.save(fp2, "JPEG")
        fp2.close()

    #img_size = ???
    #point x,y = ???
    img_size = 100
    mask = np.zeros(img_size)


    return mask


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    #Region Of Interest
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor =FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model

def get_transform(train):
    transforms = []
    #T should be changed
    if train:
        transforms.append(T.Grayscale(3))
        transforms.append(T.ToTensor())
        transforms.append(T.RandomErasing())
    else:
        transforms.append(T.ToTensor())
    #randomaffine,randomgrayscale,randomerasing
    return T.Compose(transforms)

def change_name(path,num):
    imgs = list(sorted(os.listdir(path)))
    for i in range(len(imgs)):
        os.rename(path + imgs[i],path + str(num)+imgs[i])


def temp():
    # simple check for dataset
    # test_PFD = PennFudanDataset('PennFudanPed')
    # test_PFD.__getitem__(0)

    root = 'image'
    # set data
    dataset = ViennaDataset(root, get_transform(train=True))
    #dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = ViennaDataset(root, get_transform(train=False))
    #dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    # can error: try num_workers=0
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
    )

    # for GPU. if you use CPU, remove below lines
    torch.cuda.get_device_name(0)
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #define classes
    num_classes = 3
    labels = ['background','woman','crown']
    # load model. if none, train.
    if (os.path.isfile(root+'_model.pt')):
        model = torch.load(root+'_model.pt', map_location=device)
        #model = get_instance_segmentation_model(num_classes)
        """model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        num_epochs = 20

        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            torch.cuda.empty_cache()
            #evaluate(model, data_loader_test, device=device)"""
    else:
        model = get_instance_segmentation_model(num_classes)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        num_epochs = 10

        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            torch.cuda.empty_cache()
            #evaluate(model, data_loader_test, device=device)

    # save
    torch.save(model, root+'_model.pt')

    # test
    print(len(dataset_test))
    for i in range(len(dataset_test)):
        img, _ = dataset_test[i]
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])

        im1 = Image.fromarray(img.mul(200).permute(1, 2, 0).byte().numpy())

        # draw bounding boxes & mask layer & label
        num_boxes = len(prediction[0]['boxes'])
        colors = pkl.load(open("pallete", "rb"))
        im1_ = ImageDraw.Draw(im1)
        for j in range(num_boxes):
            if prediction[0]['scores'][j].cpu().numpy() < 0.6 :
                continue
            c0 = prediction[0]['boxes'][j].cpu().numpy()
            c1 = tuple(c0[:2])
            c2 = tuple(c0[2:])
            im2 = Image.fromarray(prediction[0]['masks'][j, 0].mul(205).byte().cpu().numpy())
            color = random.choice(colors)
            layer = Image.new('RGB', im2.size, color)
            # bounding box
            im1_.rectangle((c1, c2), outline=color)
            # label
            str_ = labels[prediction[0]['labels'][j].cpu().numpy()]
            score_ = prediction[0]['scores'][j].cpu().numpy()
            score = np.round(score_, 4)
            im1_.text(c1, str_ + " score:" + str(score), fill=(255, 255, 255, 255))
            # mask layer
            im1.paste(layer, (0, 0), im2)

        fp1 = open('results/'+str(i)+'_detection.jpg', 'w')
        im1.save(fp1, "JPEG")

        #fp2 = open('results/'+str(i)+'_mask.jpg', 'w')
        #im2.save(fp2, "JPEG")
        fp1.close()
        #fp2.close()



if __name__ == '__main__':
    temp()
    #change_name("image/crown_woman/mask/",2)
    #json_path = 'image/0203/0203mask.json'
    #img_path = 'image/0203/'
    #save_path = 'image/0203/mask/'
    #if not os.path.isdir(save_path):
    #    os.mkdir(save_path)
    #save_mask_from_json(json_path,img_path,save_path)
