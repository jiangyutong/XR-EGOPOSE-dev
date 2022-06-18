# -*- coding: utf-8 -*-
"""
change the xr-egopose dataset to MPII format,
for 2D keypoints detection
the skeleton still keep mocap index 

@author: Zhang Yang

"""
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger
from utils import evaluate, io
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
LOGGER = ConsoleLogger("Main")
import json
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
'''''''''
Skeleton index:
                        5 head
                        4 neck
                10      3      6
                11      2      7
                12      1      8
                13      0      9
                     18  14
                     19  15
                     20  16
                     21  17
'''''''''

def show2Dpose(kps, img):
    # 2D connections
    connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                   [3, 6], [6, 7], [7, 8], [8, 9], 
                   [3, 10], [10, 11], [11, 12], [12, 13], 
                   [0, 14], [14, 15], [15, 16], [16, 17], 
                   [0, 18], [18, 19], [19, 20], [20, 21]]
    # left or right 
    LR = np.array( [0, 0, 0, 0, 0,
                    0, 0, 0, 0, 
                    1, 1, 1, 1, 
                    0, 0, 0, 0,
                    1, 1, 1, 1], dtype=bool)
    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), rcolor if LR[j] else lcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img

def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor='b'
    rcolor='r'

    I = np.array( [0, 1, 2, 3, 4, 3, 6, 7, 8,  3, 10, 11, 12, 0,  14, 15, 16, 0,  18,19,20])
    J = np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20,21])
    LR = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  1,  1,  1,  0,  0,  0,  0,  1, 1 , 1,1], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = rcolor if LR[i] else lcolor)

    ax.set_xlim3d([1, -1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([2, 0])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = True)
    ax.tick_params('y', labelleft = True)
    ax.tick_params('z', labelleft = True)

def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)

def img2video(video_path, output_dir):  
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])
    videoWrite = cv2.VideoWriter(video_path, fourcc, 25,size) 
    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)
    videoWrite.release()

def main():

    LOGGER.info('Starting show data visualization...')

    # ------------------- Data loader -------------------
    # let's load data from validation set as example
    data = Mocap(
        config.dataset.val,
        SetType.VAL,
        transform=None)
    data_loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False)

    prefix = "val"

    output_dir_2D = config.output_dir +'images/'
    os.makedirs(output_dir_2D, exist_ok=True)

    output_dir_show = config.output_dir +'show/'
    os.makedirs(output_dir_show, exist_ok=True)

    output_dir_annot = config.output_dir +'annot/'
    os.makedirs(output_dir_annot, exist_ok=True)

    
    print('\nGenerating  images...')

    json_file = open(output_dir_annot + prefix+'.json', 'w')
    all_labels=[]
    for it, (img, p2d, p3d, action,_,_,_) in enumerate(data_loader):
        if it%10 != 0:
            continue
 
        # get image
        img = img.numpy()[0]*255.0
        img = img.astype(np.uint8)[:,:,::-1]
        img = np.ascontiguousarray(img)
        roi = img[:,240:1040,:]
        resized_length = 256
        resized = cv2.resize(roi, (resized_length, resized_length))
        ratio = resized_length/800

        # show 2d keypoints
        p2d_np = p2d.numpy()[0]
        p2d_np[:,0] -= 240
        p2d_np *=ratio
        # 2d数据中存在标注错误，特别是点09点（右手）和13点(左手)，根据8-9和12-13骨骼的长度限制进行一次筛选
        r_hand_len= np.sqrt( (p2d_np[8][0] - p2d_np[9][0])**2 + (p2d_np[8][1] - p2d_np[9][1])**2 ) 
        l_hand_len= np.sqrt( (p2d_np[12][0] - p2d_np[13][0])**2 + (p2d_np[12][1] - p2d_np[13][1])**2 ) 
        if r_hand_len>resized_length/2 or l_hand_len>resized_length/2:
            continue
        
        # save images
        img_name = prefix + str(('%08d'% it))
        cv2.imwrite(output_dir_2D + img_name + '.png', resized)

        # # save show image 
        resized_show = resized.copy()
        resized_show =  show2Dpose(p2d_np, resized_show)
        cv2.imwrite(output_dir_show +img_name + '_2D.png', resized_show)
        
        dicts = {}
        dicts['joints']= p2d_np.tolist()
        dicts['joints_vis']= np.ones(len(p2d_np[:,0])).tolist()
        dicts['image'] = img_name + '.png'
        dicts['scale'] = 1.28
        dicts['center'] = [128,128]
        all_labels.append(dicts)

        LOGGER.info('Saved: {}'.format(it))

        if it > 10000*10:
            break
    
    json.dump(all_labels,json_file)
    LOGGER.info('Done.')

if __name__ == "__main__":
    main()
