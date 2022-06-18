# -*- coding: utf-8 -*-
"""
可视化 xr-egopose 数据集中的标注信息

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

def show3Dpose_world(vals, ax):
    ax.view_init(elev=0, azim=-40)

    lcolor='b'
    rcolor='r'

    I = np.array( [0, 1, 2, 3, 4, 3, 6, 7, 8,  3, 10, 11, 12, 0,  14, 15, 16, 0,  18,19,20])
    J = np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20,21])
    LR = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  1,  1,  1,  0,  0,  0,  0,  1, 1 , 1,1], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, z, y, lw=2, color = rcolor if LR[i] else lcolor)

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([0, 2])
    ax.set_aspect('auto')

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


    output_dir_2D = config.output_dir +'pose2D/'
    os.makedirs(output_dir_2D, exist_ok=True)
    output_dir_3D = config.output_dir +'pose3D/'
    os.makedirs(output_dir_3D, exist_ok=True)
    output_dir_pose = config.output_dir +'pose/'
    os.makedirs(output_dir_pose, exist_ok=True)
    
    print('\nGenerating  images...')
    for it, (img, p2d, p3d, p3d_w, action,subject,env) in enumerate(data_loader):

        LOGGER.info('Iteration: {}'.format(it))

        # get image
        img = img.numpy()[0]*255.0
        img = img.astype(np.uint8)[:,:,::-1]
        img = np.ascontiguousarray(img)

        # show 2d keypoints
        p2d_np = p2d.numpy()[0]
        p2d_img =  show2Dpose(p2d_np, img)
        cv2.imwrite(output_dir_2D + str(('%04d'% it)) + '_2D.png', p2d_img)

        # show 3d keypoints
        p3d_np = p3d_w.numpy()[0]
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose_world(p3d_np, ax)
        plt.savefig(output_dir_3D + str(('%04d'% it)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
        plt.close()

        if it == 300:
            break

    ## merge 2d and 3d images
    print('\nGenerating merged images...')
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])
        ## show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("2D GT", fontsize = font_size)
        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("3D GT", fontsize = font_size)
        ## save
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')

    # generate video
    video_path = config.output_dir + "data.mp4"
    img2video(video_path, config.output_dir)
    LOGGER.info('Done.')

if __name__ == "__main__":
    main()
