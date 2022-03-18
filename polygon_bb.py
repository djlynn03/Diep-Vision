import os
import random
import math
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np

import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

images_path = Path('./all_images/fullpage')
anno_path = Path('./all_images/annotations')

def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]
    
def generate_train_df (anno_path):
    annotations = filelist(anno_path, '.xml')
    anno_list = []
    for anno_path in annotations:
        root = ET.parse(anno_path).getroot()
        anno = {}
        anno['filename'] = Path(str(images_path) + '/'+ root.find("./filename").text)
        anno['width'] = root.find("./size/width").text
        anno['height'] = root.find("./size/height").text
        anno['class'] = root.find("./object/name").text
        
        anno['xmins'] = []
        anno['ymins'] = []
        anno['xmaxs'] = []
        anno['ymaxs'] = []
        
        xmins = root.findall("./object/bndbox/xmin")
        for xmin in xmins:
            anno['xmins'].append(xmin.text)
            
        ymins = root.findall("./object/bndbox/ymin")
        for ymin in ymins:
            anno['ymins'].append(ymin.text)
            
        xmaxs = root.findall("./object/bndbox/xmax")
        for xmax in xmaxs:
            anno['xmaxs'].append(xmax.text)
            
        ymaxs = root.findall("./object/bndbox/ymax")
        for ymaxs in ymaxs:
            anno['ymaxs'].append(ymaxs.text)

        # anno['xmin'] = int(root.find("./object/bndbox/xmin").text)
        # anno['ymin'] = int(root.find("./object/bndbox/ymin").text)
        # anno['xmax'] = int(root.find("./object/bndbox/xmax").text)
        # anno['ymax'] = int(root.find("./object/bndbox/ymax").text)
        print(anno)
        anno_list.append(anno)
    return pd.DataFrame(anno_list)

df_train = generate_train_df(anno_path)
#label encode target
class_dict = {'square': 0, 'triangle': 1, 'pentagon': 2, 'player': 3, 'bullet': 4, 'enemy': 5}
df_train['class'] = df_train['class'].apply(lambda x:  class_dict[x])

print(df_train.shape)
df_train.head()

def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    for i in range(len(bb[0])): # [100, 200, 300] xmins
        Y[bb[0][i]:bb[2][i], bb[1][i]:bb[3][i]] = 1.

    return Y

def create_masks(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    Ys = []
    rows,cols,*_ = x.shape
    bb = bb.astype(np.int)
    for i in range(len(bb[0])): # [100, 200, 300] xmins
        Y = np.zeros((rows, cols))
        Y[bb[0][i]:bb[2][i], bb[1][i]:bb[3][i]] = 1.
        Ys.append(Y)
    return Ys

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def masks_to_bb(Y_arr):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    returnv = []
    for Y in Y_arr:
        cols, rows = np.nonzero(Y)
        if len(cols)==0:
            returnv.append(np.zeros(4, dtype=np.float32))
        else:
            returnv.append(np.array([np.min(cols), np.min(rows), np.max(cols), np.max(rows)], dtype=np.float32))
            
    # 0000000000001110000
    # 0011100000001110000
    # 0011100000000000000
    # 0000000000001110000
    # 0000110000001110000
    # 0000110000000000000
    
    # top_row = np.min(rows)
    # left_col = np.min(cols)
    # bottom_row = np.max(rows)
    # right_col = np.max(cols)
    return np.array(returnv, dtype=np.float32)


def create_bb_array(x):
    """Generates bounding box array from a train_df row"""    
    return np.array([x[5],x[4],x[7],x[6]])  #ymin, xmin, ymax, xmax

def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    
    arr = [cv2.resize(i, (int(1.49*sz), sz)) for i in create_masks(bb, im)]
    return new_path, masks_to_bb(arr)

new_paths = []
new_bbs = []
train_path_resized = Path('./all_images/images_resized')
for index, row in df_train.iterrows():
    new_path,new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values),300)
    new_paths.append(new_path)
    new_bbs.append(new_bb)
df_train['new_path'] = new_paths
df_train['new_bb'] = new_bbs

# im = cv2.imread(str(df_train.values[2][0]))
# bb = create_bb_array(df_train.values[2])
# print(im.shape)
# Y = create_mask(bb, im)
# mask_to_bb(Y)
# plt.imshow(im)
# plt.imshow(Y, cmap='gray')

# plt.show()

# modified from fast.ai
def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)

def create_corner_rect(bb, color='green'):
    print(bb)
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=1)

def show_corner_bb(im, bb):
    plt.imshow(im)
    for i in range(len(bb)):
        plt.gca().add_patch(create_corner_rect(bb[i]))
    plt.show()
    
im = cv2.imread(str(df_train.values[1][8]))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
show_corner_bb(im, df_train.values[1][9])

df_train = df_train.reset_index()
X = df_train[['new_path', 'new_bb']]
Y = df_train['class']
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]
class ShapeDataset(Dataset):
    def __init__(self, paths, bb, y, transforms=False):
        self.transforms = transforms
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = self.y[idx]
        x, y_bb = transformsXY(path, self.bb[idx], self.transforms)
        x = normalize(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb
    
train_ds = ShapeDataset(X_train['new_path'],X_train['new_bb'] ,y_train, transforms=True)
valid_ds = ShapeDataset(X_val['new_path'],X_val['new_bb'],y_val)
batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)

class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)
    
def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr
def train_epocs(model, optimizer, train_dl, val_dl, epochs=10,C=1000):
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_class, y_bb in train_dl:
            batch = y_class.shape[0]
            x = x.cpu().float()
            y_class = y_class.cpu()
            y_bb = y_bb.cpu().float()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        val_loss, val_acc = val_metrics(model, valid_dl, C)
        print("epoch %f train_loss %.3f val_loss %.3f val_acc %.3f" % (idx, train_loss, val_loss, val_acc))
    return sum_loss/total

def val_metrics(model, valid_dl, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    for x, y_class, y_bb in valid_dl:
        batch = y_class.shape[0]
        x = x.cpu().float()
        y_class = y_class.cpu()
        y_bb = y_bb.cpu().float()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb/C
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    return sum_loss/total, correct/total

model = BB_model().cpu()
model.load_state_dict(torch.load('image_model_weights.pt'))
# model.eval()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.006)
# train_epocs(model, optimizer, train_dl, valid_dl, epochs=200)

# torch.save(model.state_dict(), 'image_model_weights.pt')

im = read_image('./all_images/images_resized/page1.png')
im = cv2.resize(im, (int(1.49*300), 300))
cv2.imwrite('./all_images/shapes_test/page1.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

test_ds = ShapeDataset(
        pd.DataFrame([{'path':'./all_images/shapes_test/page1.png'}])['path'],
        pd.DataFrame([{'bb':np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])}])['bb'],
        pd.DataFrame([{'y':[0,0,0,0,0,0,0]}])['y']
    )

# test_ds = ShapeDataset(X_train['new_path'],X_train['new_bb'] ,y_train, transforms=True)
x, y_class, y_bb = test_ds[0]

xx = torch.FloatTensor(x[None,])
xx.shape

model.load_state_dict(torch.load('image_model_weights.pt'))
model.eval()

out_class, out_bb = model(xx.cpu())
out_class, out_bb

print(torch.max(out_class, 1))

bb_hat = out_bb.detach().cpu().numpy()
bb_hat = bb_hat.astype(int)
show_corner_bb(im, bb_hat)