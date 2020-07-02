#written by mohomin123@gmail.com
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

from PIL import Image
import os
import numpy as np

#%matplotlib inline
torch.cuda.get_device_name(0)
"""
#prepare batch_loader
trn_dataset = datasets.MNIST('../mnist_data/', download=True,
                             train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,),(0.3081,))
                             ])
                             )
val_dataset = datasets.MNIST('../mnist_data/', download=False,
                             train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,),(0.3081,))
                             ])
                             )
"""
#prepare dataset
#Image.open('PennFudanPed/PNGImages/FudanPed00001.png')
#mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
#mask.putpalette([
#    0,0,0,
#    255,0,0,
#    255,255,0,
#    255,153,0
#])

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
        target["lables"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


test_PFD = PennFudanDataset('PennFudanPed')
test_PFD.__getitem__(0)

"""
batch_size = 64
trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

use_cuda = torch.cuda.is_available()


class CNNClassifier(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNNClassifier, self).__init__()
        conv1 = nn.Conv2d(1, 6, 5, 1)  # 6@24*24
        # activation ReLU
        pool1 = nn.MaxPool2d(2)  # 6@12*12
        conv2 = nn.Conv2d(6, 16, 5, 1)  # 16@8*8
        # activation ReLU
        pool2 = nn.MaxPool2d(2)  # 16@4*4

        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool1,
            conv2,
            nn.ReLU(),
            pool2
        )

        fc1 = nn.Linear(16 * 4 * 4, 120)
        # activation ReLU
        fc2 = nn.Linear(120, 84)
        # activation ReLU
        fc3 = nn.Linear(84, 10)

        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)  # @16*4*4
        # make linear
        dim = 1
        for d in out.size()[1:]:  # 16, 4, 4
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return F.softmax(out, dim=1)

cnn = CNNClassifier()

# loss
criterion = nn.CrossEntropyLoss()
# backpropagation method
learning_rate = 1e-3
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# hyper-parameters
num_epochs = 10
num_batches = len(trn_loader)

trn_loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    trn_loss = 0.0
    for i, data in enumerate(trn_loader):
        x, label = data
        if use_cuda:
            x = x.cuda()
            label = label.cuda()
        # grad init
        optimizer.zero_grad()
        # forward propagation
        model_output = cnn(x)
        # calculate loss
        loss = criterion(model_output, label)
        # back propagation
        loss.backward()
        # weight update
        optimizer.step()

        # trn_loss summary
        trn_loss += loss.item()
        # del (memory issue)
        del loss
        del model_output

        # 학습과정 출력
        if (i + 1) % 100 == 0:  # every 100 mini-batches
            with torch.no_grad():  # very very very very important!!!
                val_loss = 0.0
                for j, val in enumerate(val_loader):
                    val_x, val_label = val
                    if use_cuda:
                        val_x = val_x.cuda()
                        val_label = val_label.cuda()
                    val_output = cnn(val_x)
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss

            print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format(
                epoch + 1, num_epochs, i + 1, num_batches, trn_loss / 100, val_loss / len(val_loader)
            ))

            trn_loss_list.append(trn_loss / 100)
            val_loss_list.append(val_loss / len(val_loader))
            trn_loss = 0.0
"""