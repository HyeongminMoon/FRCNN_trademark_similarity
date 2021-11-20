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

from dataset import ViennaDataset, PennFudanDataset
"""
data에는 __getitem__과 __len__이 있어야 함.
__getitem__은 image, target을 반환해야 함.
__len__은 이미지의 길이를 반환.
"""
def cos_sim(A, B):
    device = torch.device('cuda')
    A = torch.from_numpy(A).to(device)
    B = torch.from_numpy(B).to(device)
    return F.cosine_similarity(A,B)

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
        #draw masks
        img = Image.open(img_path+img_name)
        mask = np.zeros((img.size[1],img.size[0]))
        for i in range(len(mask_regions)):
            region_points = mask_regions[i]['shape_attributes']
            all_points_x = region_points['all_points_x']
            all_points_y = region_points['all_points_y']
            rr, cc = skimage.draw.polygon(all_points_y,all_points_x)
            mask[rr,cc] = i+1
        # save mask

        png_img = png.from_array(mask, mode="L;16")
        png_img.save(save_path+img_name.replace('.jpg', '_mask.png'))
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


def extract_code(input_id):
    if not os.path.isdir(str(input_id)):
        os.mkdir(str(input_id))
    #input_id = 12524
    shutil.rmtree('../../input/')
    os.mkdir('../../input/')
    shutil.copy('../../images/img_' + str(input_id) + '.jpg', '../../input/img_' + str(input_id) + '.jpg')
    root = '../../images'


    dataset_input = ViennaDataset("../../input", get_transform(train=False), test_mode = True)
    #dataset_test = ViennaDataset(root, get_transform(train=False), test_mode = True)
    #dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    torch.manual_seed(1)
    # can error: try num_workers=0
    #data_loader_test = torch.utils.data.DataLoader(
    #    dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
    #)

    # for GPU. if you use CPU, remove below lines
    torch.cuda.get_device_name(1)
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #define classes
    num_classes = 29
    labels = ['background', 'character', 'sun', 'planet', 'heart', 'ceramic', 'bottle', 'bowl', 'star', 'jewel',
              'moon','triangle','arrow','rect','shield','building','tree','flower','car','airplane',
              'crown','hat','cross','landscape','mountain','plantac','leaf','fruit','tool']
    
    if (os.path.isfile('image_model.pt')):
        model = torch.load('image_model.pt', map_location=device)
    else:
        print("model doesn't exist")
    # test
    shutil.rmtree(str(input_id))
    os.mkdir(str(input_id))
    #input
    img, name = dataset_input[0]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])    
    num_boxes = len(prediction[0]['boxes'])
    p = []
    num = 0
    for j in range(num_boxes):
        if prediction[0]['scores'][j].cpu().numpy() < 0.2 :
            continue
        if num >= 3: break
        str_ = labels[prediction[0]['labels'][j].cpu().numpy()]
        index = labels.index(str_)
        if index in p: continue
        p.append(index)
        num += 1
    p.sort()
    input_name = name
    input_code = p
    print(input_name, input_code)
    
    same_list = []
    sim_list = []
    correct_coef = (3 - len(input_code)) * 0.05
    
    img_list = list(sorted(os.listdir('../../npys_code/')))
    load_features = np.load('../../npys_eb7_gray.npy')
    input_feature = np.full(load_features.shape,load_features[input_id-1])#TODO

    load_features = torch.from_numpy(load_features).to(device)
    input_feature = torch.from_numpy(input_feature).to(device)

    img_sim = F.cosine_similarity(input_feature,load_features)
    for i in range(len(img_sim)):
        try:
            if img_sim[i].item() < 0.4: continue
            p = np.load("../../npys_code/" + str(i+1) + '.npy')
            #np.save("../../npys_code/" + name[4:-4], np.array(p))
            same_num = 0
            for code in input_code:
                if code in p:
                    same_num += 1           
            if same_num == 3:
                code_sim = 1.0
            elif same_num == 2:
                code_sim = 0.95
            elif same_num == 1:
                code_sim = 0.9
            else:
                code_sim = 0.8
            code_sim += correct_coef
            similarity = round(code_sim * img_sim[i].item(), 6)
            if similarity > 0.3:
                sim_list.append([str(i+1),similarity])
            print(i,end='\r')
            #path_ = "results/" + str(format(similarity,"6f"))[2:] + name
            #fp1 = open(path_, 'w')
            #im1.save(fp1, "JPEG")
            #fp1.close()
        except:
            print("except i:",i)
    sim_list = sorted(sim_list, key=lambda s : s[1], reverse=True)
    #print(sim_list)
    for i in range(min(len(sim_list),100)):
        shutil.copy('../../images/img_' + str(sim_list[i][0]) + '.jpg', str(input_id) + '/' + str(format(sim_list[i][1],"6f")) + '_' +str(sim_list[i][0]) + '.jpg')

def train_frcnn_model():
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
    t = int(len(indices)*0.2) # 20% test data
    dataset = torch.utils.data.Subset(dataset, indices[:-t])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-t:])
    # can error: try num_workers=0
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
    )

    # for GPU. if you use CPU, remove below lines
    torch.cuda.get_device_name(1)
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #define classes
    num_classes = 56
    labels = ['background','woman','crown','star','man','child','jewel','animal','sun','moon','amphibia',
              'aqua','bird','building','ceramic','dinosaur','flag','furniture','leaf','planet','plate',
              'ship','shoes','stationery','watch','wheel','airplane','anchor','atom','book','bowl',
              'circle','cross','food','hat','heart','helmat','insect','landscape','map','mountain',
              'part','plantac','rect','river','tool','tree','triangle','bottle','fruit','phone',
              'flower','arrow','car','ribbon','shield']
    num_classes = 29
    labels = ['background', 'character', 'sun', 'planet', 'heart', 'ceramic', 'bottle', 'bowl', 'star', 'jewel',
              'moon','triangle','arrow','rect','shield','building','tree','flower','car','airplane',
              'crown','hat','cross','landscape','mountain','plantac','leaf','fruit','tool']
    num_classes = 10
    labels = ['background', 'woman','man','child','animal','amphibia','aqua','bird','dinosaur','insect']
    num_classes = 5
    labels = ['background', 'human', 'animal', 'bird', 'insect']
    #num_classes = 14
    #labels = ['background', 'human', 'animal', 'circle', 'bowl', 'triangle', 'rect', 'tree', 'car', 'tool', 'hat', 'cross', 'part', 'landsacpe']
    #human = ['man', 'child', 'woman']
    #animal = ['animal','aqua','bird','amphibia','dinosaur','insect']
    #circle = ['sun', 'circle','heart','planet','wheel','watch','fruit']
    #bowl = ['bowl', 'plate', 'bottle', 'ceramic']
    #triangle = ['star','moon','jewel','leaf','triangle','arrow','flag']
    #rect = ['rect', 'building','book','furniture','shield']
    #tree = ['tree','flower']
    #car = ['car', 'ship', 'airplane']
    #tool = ['stationery','food','tool']
    #hat = ['hat','helmat','crown','phone']
    #cross = ['cross','anchor','atom','ribbon']
    #part = ['part', 'shoes']
    #landscape = ['landscape', 'map', 'mountain','plantac','river']


    # load model. if none, train.
    if (os.path.isfile(root+'_model.pt')):
        model = torch.load(root+'_model.pt', map_location=device)
        evaluate(model, data_loader_test, device=device)
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
        
        # save
        torch.save(model, root+'_model.pt')

    # test
    shutil.rmtree('results/')
    os.mkdir('results')

    print(len(dataset_test))
    for i in range(len(dataset_test)):
        if i == 302 or i == 601 or i == 723 : continue
        torch.cuda.empty_cache()
        img, _ = dataset_test[i]
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])
        #print(prediction)
        im1 = Image.fromarray(img.mul(200).permute(1, 2, 0).byte().numpy())
        
        #print(prediction[0]['scores'])
        #print(prediction[0]['labels'])
        # draw bounding boxes & mask layer & label
        num_boxes = len(prediction[0]['boxes'])
        colors = pkl.load(open("pallete", "rb"))
        im1_ = ImageDraw.Draw(im1)
        
        #top 3
        predicted_labels = ''
        p = []
        num = 0
        for j in range(num_boxes):
            if prediction[0]['scores'][j].cpu().numpy() < 0.2 :
                continue
            if num >= 3: break
            str_ = labels[prediction[0]['labels'][j].cpu().numpy()]
            if str_ in p: continue
            c0 = prediction[0]['boxes'][j].cpu().numpy()
            c1 = tuple(c0[:2])
            c2 = tuple(c0[2:])
            im2 = Image.fromarray(prediction[0]['masks'][j, 0].mul(100).byte().cpu().numpy())
            color = random.choice(colors)
            layer = Image.new('RGB', im2.size, color)
            # bounding box
            im1_.rectangle((c1, c2), outline=color)
            # label
            predicted_labels = predicted_labels + '_' + str_
            score_ = prediction[0]['scores'][j].cpu().numpy()
            score = np.round(score_, 4)
            im1_.text(c1, str_ + " score:" + str(score), fill=(255, 255, 255, 255))
            # mask layer
            im1.paste(layer, (0, 0), im2)
            p.append(str_)
            num += 1

        fp1 = open('results/'+str(i)+'_detection'+predicted_labels+'.jpg', 'w')
        im1.save(fp1, "JPEG")

        #fp2 = open('results/'+str(i)+'_mask.jpg', 'w')
        #im2.save(fp2, "JPEG")
        fp1.close()
        #fp2.close()

def evaluate_frcnn_model():
    
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.004, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    num_epochs = 15

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        lr_scheduler.step()

if __name__ == '__main__':
    train_frcnn_model()
    evaluate_frcnn_model()
    
    
