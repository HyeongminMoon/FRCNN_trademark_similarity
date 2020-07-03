#written by mohomin123@gmail.com
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pickle as pkl
import random
import cv2

from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import os
import numpy as np

#copied from torchvision
from engine import train_one_epoch, evaluate
import utils
import transforms as T

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
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == '__main__':
    # simple check for dataset
    # test_PFD = PennFudanDataset('PennFudanPed')
    # test_PFD.__getitem__(0)


    # set data
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

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
    num_classes = 2
    labels = ['background','person']
    # load model. if none, train.
    if (os.path.isfile('./model.pt')):
        model = torch.load('./model.pt', map_location=device)
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
            evaluate(model, data_loader_test, device=device)

    # save
    torch.save(model, './model.pt')

    # test
    img, _ = dataset_test[12]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    im1 = Image.fromarray(img.mul(200).permute(1, 2, 0).byte().numpy())

    # draw bounding boxes & mask layer & label
    num_boxes = len(prediction[0]['boxes'])
    colors = pkl.load(open("pallete","rb"))
    im1_ = ImageDraw.Draw(im1)
    print(prediction[0]['boxes'])
    for i in range(num_boxes):
        c0 = prediction[0]['boxes'][i].cpu().numpy()
        print(c0)
        c1 = tuple(c0[:2])
        c2 = tuple(c0[2:])
        im2 = Image.fromarray(prediction[0]['masks'][i, 0].mul(205).byte().cpu().numpy())
        color = random.choice(colors)
        layer = Image.new('RGB', im2.size, color)
        #bounding box
        im1_.rectangle((c1, c2), outline=color)
        #label
        str_ = labels[prediction[0]['labels'][i].cpu().numpy()]
        score_ = prediction[0]['scores'][i].cpu().numpy()
        score = np.round(score_,4)
        im1_.text(c1, str_+" score:"+ str(score),fill=(255,255,255,255))
        #mask layer
        im1.paste(layer, (0,0), im2)


    fp1 = open('im1.jpg','w')
    im1.save(fp1,"JPEG")

    fp2 = open('im2.jpg','w')
    im2.save(fp2,"JPEG")
    fp1.close()
    fp2.close()
