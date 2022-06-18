# -*- coding: utf-8 -*-
"""
测试鱼眼相机参数

@author: Zhang Yang

"""

import numpy as np
import json 
import cv2
import transformations
def camera_world_to_transform(path):

    f = open(path)

    data = json.load(f)


    ## 鱼眼相机内参
    Khmc = np.array([ [352.59619801644876, 0.0, 0.0],
                      [0.0, 352.70276325061578, 0.0],
                      [654.6810228318458, 400.952228031277, 1.0] ]).T
    kd = np.array([-0.05631891929412012, -0.0038333424842925286, -0.00024681888617308917, -0.00012153386798050158])
    Mmaya = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
    h_fov = np.array(data['camera']['cam_fov'])


    #相机外参
    translation = np.array(data['camera']['trans'])
    rotation = np.array(data['camera']['rot']) * np.pi / 180.0
    Mf =  transformations.euler_matrix(rotation[0],
                                      rotation[1],
                                      rotation[2],
                                      'sxyz')
    Mf[0:3, 3] = translation
    Mf = np.linalg.inv(Mf)
    #M为相机的旋转和平移矩阵
    M = Mmaya.T.dot(Mf)

    # 世界坐标系的位置
    joints = np.vstack([j['trans'] for j in data['joints']]).T

    # 世界坐标到相机坐标系
    Xj = M[0:3, 0:3].dot(joints) + M[0:3, 3:4]

    # 相机坐标系下的3d坐标位置
    pts3d_json = data["pts3d_fisheye"]
    print(np.allclose(Xj,pts3d_json))

    # 从相机坐标系返回世界坐标系
    M_inv = np.linalg.inv(M[0:3, 0:3])
    joints_ = M_inv.dot((pts3d_json - M[0:3, 3:4]))
    print(np.allclose(joints,joints_))

    # 根据相机内参、得到相机坐标系下的2D坐标位置
    pts2d, jac = cv2.fisheye.projectPoints(
        Xj.T.reshape((1, -1, 3)).astype(np.float32),
        (0, 0, 0),
        (0, 0, 0),
        Khmc,
        kd
    )
    pts2d = pts2d.T.reshape((2, -1))
    pts2d_json = data["pts2d_fisheye"]
    # print(np.allclose(pts2d,pts2d_json))


camera_world_to_transform("/media/zlz422/jyt/egpose/XR-EGOPOSE/data/ValSet/male_008_a_a/env_002/cam_down/json/male_008_a_a_000008.json")

