# -*- coding: utf-8 -*-
"""
change the xr-egopose dataset to hm36 format,
for 3D keypoints detection
the skeleton will change to hm36 format 

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
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib
matplotlib.use('TkAgg')
from skimage import io as sio
import matplotlib.pyplot as plt
'''''''''
egopose index:
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

'''''''''
hm36 index_db
                        10 
                        9 
                14      8      11
                15      7      12
                16      0      13 
                     1     4
                     2     5
                     3     6
'''''''''

Mopcap2hm36=[0,-1,7,8,9,10,-1,11,12,13,-1,14,15,16,4,5,-1,6,1,2,-1,3]

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

def show2Dpose_hm36(kps, img):
    # 2D connections
    connections = [[0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], 
                   [8, 14], [14, 15], [15, 16], 
                   [0, 4], [4, 5], [5, 6], 
                   [0, 1], [1, 2], [2, 3]]
    # left or right 
    LR = np.array( [0, 0, 0, 0,
                    0, 0, 0, 
                    1, 1, 1, 
                    0, 0, 0,
                    1, 1, 1], dtype=bool)
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

    ax.set_xlim3d([-2, 0])
    ax.set_ylim3d([0, 2])
    ax.set_zlim3d([2, 0])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = True)
    ax.tick_params('y', labelleft = True)
    ax.tick_params('z', labelleft = True)

def show3Dpose_hm36(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor='b'
    rcolor='r'

    I = np.array( [0, 7, 8, 9, 8, 11, 12, 8, 14,  15, 0, 4, 5, 0,  1, 2])
    J = np.array( [7, 8, 9, 10,11, 12,13,14, 15, 16, 4, 5, 6, 1, 2, 3])

    LR = np.array([0, 0, 0, 0,
                    0, 0, 0, 
                    1, 1, 1, 
                    0, 0, 0,
                    1, 1, 1], dtype=bool)
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = rcolor if LR[i] else lcolor)

    ax.set_xlim3d([1, -1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([2, 0])
    ax.view_init(141,-91)
    ax.invert_yaxis()
    # ax.set_xlim3d([0, -1])
    # ax.set_ylim3d([0, 1])
    # ax.set_zlim3d([0, 0.3])
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
    names = sorted(glob.glob(os.path.join(output_dir, '*_2D.png')),key=lambda s:int(s[-10:-7]))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])
    videoWrite = cv2.VideoWriter(video_path, fourcc, 25,size) 
    for name in names:
        print(name)
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

    prefix = "Val"
    output_dir = "egopose_hm36_trainval/val/"
    os.makedirs(output_dir, exist_ok=True)

    save_idx = 0

    all_data={}
    for it, (img, p2d, p3d, p3d_w, action,subject,env,img_path,cam) in enumerate(data_loader):
        if subject[0]== "female_004_a_a":
            print("ok")
        # dir_name = "/media/zlz422/jyt/egpose/imagedata"+img_path[0][35:61]
        # if not os.path.exists(dir_name):  # os模块判断并创建
        #     os.makedirs(dir_name)
        # img = sio.imread(img_path[0]).astype(np.float32)
        # res = cv2.resize(img, (400, 400))
        # save_image_path=dir_name+img_path[0][83:]
        # cv2.imwrite(save_image_path, res)
        print(subject)
        # show 2d keypoints
        resized_length = 256
        ratio = resized_length/800
        p2d_np = p2d.numpy()[0]
        p2d_np[:,0] -= 240
        p2d_np *=ratio
        # 2d数据中存在标注错误，特别是点09点（右手）和13点(左手)，根据8-9和12-13骨骼的长度限制进行一次筛选
        r_hand_len= np.sqrt( (p2d_np[8][0] - p2d_np[9][0])**2 + (p2d_np[8][1] - p2d_np[9][1])**2 )
        l_hand_len= np.sqrt( (p2d_np[12][0] - p2d_np[13][0])**2 + (p2d_np[12][1] - p2d_np[13][1])**2 )
        if r_hand_len>resized_length/2 or l_hand_len>resized_length/2:
            continue
        img_name = prefix + str(('%08d'% save_idx))

        #convert2dindex
        p2d_h36=np.zeros((17,2))
        for mocap_idx in range(22):
            h36_idx = Mopcap2hm36[mocap_idx]
            if h36_idx != -1:
                p2d_h36[h36_idx] = p2d_np[mocap_idx]

        # save show image
        # if save_idx <=800:
        #     img = img.numpy()[0]*255.0
        #     img = img.astype(np.uint8)[:,:,::-1]
        #     img = np.ascontiguousarray(img)
        #     roi = img[:,240:1040,:]
        #     resized = cv2.resize(roi, (resized_length, resized_length))
        #     resized_show = resized.copy()
        #     resized_show =  show2Dpose_hm36(p2d_h36, resized_show)
        #     cv2.imwrite(output_dir +img_name + '_2D.png', resized_show)

        # show 3d keypoints
        p3d_np = p3d.numpy()[0]
        p3dw_np = p3d_w.numpy()[0]
        #convert2dindex
        p3d_h36=np.zeros((17,3))
        for mocap_idx in range(22):
            h36_idx = Mopcap2hm36[mocap_idx]
            if h36_idx != -1:
                p3d_h36[h36_idx] = p3d_np[mocap_idx]
        p3d_h36w = np.zeros((17, 3))
        for mocap_idx in range(22):
            h36_idx = Mopcap2hm36[mocap_idx]
            if h36_idx != -1:
                p3d_h36w[h36_idx] = p3dw_np[mocap_idx]
        # if save_idx <=800:
        #     fig = plt.figure(figsize=(9.6, 5.4))
        #     gs = gridspec.GridSpec(1, 1)
        #     gs.update(wspace=-0.00, hspace=0.05)
        #     # ax = plt.subplot(111, projection='3d')
        #     ax = plt.subplot(gs[0], projection='3d')
        #     show3Dpose_hm36(p3d_h36w, ax)
        #     # plt.show()
        #     plt.savefig(output_dir +img_name + '_3Dw.png', dpi=200, format='png', bbox_inches = 'tight')
        #     plt.close()

        # 准备生成dict数据
        subject = subject[0]
        env = env[0]
        if not subject in all_data.keys():
            all_data[subject]={}
            all_data[subject][env]={}
            all_data[subject][env]["positions"]=[]
            all_data[subject][env]["positions_3d"]=[]
            all_data[subject][env]["image_path"] = []
        else:
            if not env in all_data[subject].keys():
                all_data[subject][env]={}
                all_data[subject][env]["positions"]=[]
                all_data[subject][env]["positions_3d"]=[]
                all_data[subject][env]["image_path"] = []
            else:
                pass

        all_data[subject][env]["positions"].append(p2d_h36)
        all_data[subject][env]["positions_3d"].append(p3d_h36)
        all_data[subject][env]["image_path"].append(img_path)

        LOGGER.info('Saved: {}'.format(it))

        save_idx+=1

        # if it > 10:
        #     break

    np.savez_compressed(output_dir + prefix + "_data",egopose=all_data)

    LOGGER.info('Done.')

def test(f):
    alldatas = np.load(f,allow_pickle=True)
    pass

def merge(f1,f2):
    data1 = np.load(f1,allow_pickle=True)
    data2 = np.load(f2,allow_pickle=True)
    data1 = data1["egopose"].item()
    data2 = data2["egopose"].item()
    print(data1.keys())
    for key in data2.keys():
        data1[key] = data2[key]

    print(data1.keys())

    np.savez_compressed("egopose_hm36_trainval/merged",egopose=data1)
    


if __name__ == "__main__":
    main()
    # img2video('/media/zlz422/jyt/egpose/XR-EGOPOSE-dev/XR-EGOPOSE-dev/resultval2d.mp4','/media/zlz422/jyt/egpose/XR-EGOPOSE-dev/XR-EGOPOSE-dev/egopose_hm36_trainval/val/')
    # test("egopose_hm36_trainval/val_data.npz")
    # merge("egopose_hm36_trainval/val_data.npz","egopose_hm36_trainval/train_data.npz")
