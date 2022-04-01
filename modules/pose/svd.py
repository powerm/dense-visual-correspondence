from sre_constants import SUCCESS
import numpy as np 
import cv2 as cv 
import math 
import random

from utils.constants import DEPTH_IM_RESCALE 

############# 世界坐标系的3d点#############
objp = np.zeros((10*10,3), np.float32)
objp[:, :2]= np.mgrid[0:200:20, 0:200:20].T.reshape(-1,2)

####################相机内参#############
f=8
dx =0.01
dy=0.01
u0=320
v0=240
list1=[f/dx,0,u0,0,0,f/dy,v0,0,0,0,1,0]
M1 = np.mat(list1).reshape(3,4)

################### 相机外参#############
M2 = np.array([[],[],[],[],[]])

############## 创建空白图像##############


from math import degrees as dg 
import numpy as np 
import cv2 as cv 
import glob 
import random 




def  pinhole_projection_image_to_camera_coordinates(uv, z, K):
    """
    Takes a (u,v) pixel location to it's 3D location in camera frame.
    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for a detailed explanation.

    :param uv: pixel location in image
    :type uv:
    :param z: depth, in camera frame
    :type z: float
    :param K: 3 x 3 camera intrinsics matrix
    :type K: numpy.ndarray
    :return: (x,y,z) in camera frame
    :rtype: numpy.array size (3,)
    """

    #warnings.warn("Potentially incorrect implementation", category=DeprecationWarning)
    fx =  K[0, 0]
    fy =  K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    u =   np.asarray(uv[0]).astype(np.float32)
    v =   np.asarray(uv[1]).astype(np.float32)
    X =  (u- cx)*z/fx
    Y =   (v- cy)*z/fy
    # u_v_1 =  np.ones((3,len(uv[0])))
    # np.array([uv[0], uv[1], ])
    # K_inv = np.linalg.inv(K)
    # pos = z * K_inv.dot(u_v_1)
    pos = np.vstack((X, Y, z))
    return pos

def apply_transform(vec3, transform4):
    ones_row = np.ones_like(vec3[0,:]).reshape(1,-1)
    ones_row= ones_row.astype(np.float)
    vec4 = np.vstack((vec3, ones_row))
    vec4 = np.matmul(transform4, vec4)
    return vec4[0:3]


def invert_transform(transform4):
    transform4_copy = np.copy(transform4)
    R = transform4_copy[0:3,0:3]
    R = np.transpose(R)
    transform4_copy[0:3,0:3] = R
    t = transform4_copy[0:3,3]
    inv_t = -1.0 * np.transpose(R).dot(t)
    transform4_copy[0:3,3] = inv_t
    return transform4_copy


def  rigid_transform_3D(A, B):
    # A,B shape (N,3)
    assert len(A) == len(B)
    N = A.shape[0]
    centroid_A = np.mean(A, axis =0)
    centroid_B = np.mean(B, axis =0)
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B,(N,1))
    
    H = np.matmul(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)
    
    # special reflection case 
    if np.linalg.det(R) <0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)
    
    t = -np.matmul(R, centroid_A) + centroid_B
    
    return R, t

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform(A, B):
    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)
    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t

    
    


import modules.utils.utils as utils
#utils.add_dense_correspondence_to_python_path()
#import correspondence_plotter
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
import os
import numpy as np
from  modules.utils.transformations import identity_matrix
from dense_correspondence.correspondence_tools import correspondence_finder, correspondence_plotter
import torch 

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
def  get_point3d(uv, depth, K, image_width):
    uv_flattened = uv[1].type(dtype_long)*image_width+uv[0].type(dtype_long)
    depth_torch = torch.from_numpy(depth.copy()).type(torch.FloatTensor)
    depth_torch = torch.squeeze(depth_torch, 0)
    depth_torch = depth_torch.view(-1,1)
    
    DEPTH_IM_SCALE = 1000.0 # 
    #depth_vec = torch.index_select(img_a_depth_torch, 0, uv_a_vec_flattened)*1.0/DEPTH_IM_SCALE
    depth_vec = torch.index_select(depth_torch, 0, uv_flattened)*1.0/DEPTH_IM_SCALE
    depth_vec = depth_vec.squeeze(1)
    u_vec = uv[0].type(torch.FloatTensor)*depth_vec
    v_vec = uv[1].type(torch.FloatTensor)*depth_vec
    z_vec = depth_vec
    full_vec = torch.stack((u_vec, v_vec, z_vec))
    K_inv = np.linalg.inv(K)
    K_inv_torch = torch.from_numpy(K_inv).type(torch.FloatTensor)
    point3d = K_inv_torch.mm(full_vec)
    return point3d
    
def apply_transform_torch(vec3, transform4):
    ones_row = torch.ones_like(vec3[0,:]).type(dtype_float).unsqueeze(0)
    vec4 = torch.cat((vec3,ones_row),0)
    vec4 = transform4.mm(vec4)
    return vec4[0:3]


config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'caterpillar_only.yaml')

config = utils.getDictFromYamlFilename(config_filename)
dataset = SpartanDataset(config=config)

scene = "2018-04-10-16-02-59"
img_a_index = dataset.get_random_image_index(scene)
img_a_rgb, img_a_depth, img_a_mask, img_a_pose = dataset.get_rgbd_mask_pose(scene, 0)

img_b_index = dataset.get_img_idx_with_different_pose(scene, img_a_pose, num_attempts=50)
img_b_rgb, img_b_depth, _, img_b_pose = dataset.get_rgbd_mask_pose(scene, img_b_index)

camera_intrinsics = dataset.get_camera_intrinsics(scene)
camera_intrinsics_K = camera_intrinsics.K
img_a_depth_numpy = np.asarray(img_a_depth)
img_b_depth_numpy = np.asarray(img_b_depth)

uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(img_a_depth_numpy, img_a_pose,
                                                                            img_b_depth_numpy, img_b_pose,
                                                                            img_a_mask=np.array(img_a_mask),
                                                                            num_attempts= 20,
                                                                            K=camera_intrinsics_K
                                                                            )
correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, uv_a, uv_b)
aPoint_at_camera= get_point3d(uv_a, img_a_depth_numpy,camera_intrinsics_K, 640)
aPoint_at_world = apply_transform_torch(aPoint_at_camera, torch.from_numpy(img_a_pose).type(dtype_float))

# b相对于世界坐标系的三维点
bPoint_at_camera= get_point3d(uv_b, img_b_depth_numpy,camera_intrinsics_K, 640)

img_b_pose_inv = invert_transform(img_b_pose)
Point_b_re = apply_transform_torch(aPoint_at_world, torch.from_numpy(invert_transform(img_b_pose)).type(dtype_float))

point3d_b_trans = torch.from_numpy(img_b_pose_inv[0:3, 0:3]).type(dtype_float).mm(aPoint_at_world)+ torch.from_numpy(img_b_pose_inv[0:3, 3].reshape(3,1)).type(dtype_float)
print('b-bb:', point3d_b_trans.reshape(-1,3) - bPoint_at_camera.reshape(-1,3))

R, t = rigid_transform(np.array(aPoint_at_world), np.array(bPoint_at_camera))


def isRotationMatrix(M):
    tag = False
    I = np.identity(M.shape[0])
    if np.all(np.matmul(M,M.T)== I) and (np.linalg.det(M) == 1):
        tag = True
    return tag
print(isRotationMatrix(R))

def  mse(B, B2):
    err = B2 - B 
    err = err*err
    err = np.sum(err)
    rmse = np.sqrt(err/B.shape[1])
    return rmse

print(R)
print("\n")
print(t)
print("\n")
print(invert_transform(img_b_pose))

point3d_b_pre =  R@(np.array(aPoint_at_world)) + t
#print('chongtouyin:', point3d_b_pre.reshape(-1,3) - np.array(bPoint_at_camera).reshape(-1,3))
print("mse:", mse(np.array(bPoint_at_camera), point3d_b_pre))
print("\n")

linear_distance = np.linalg.norm(t - img_b_pose[0:3, 3])
print("\n")
print(linear_distance)
linear_distance = np.linalg.norm(t - invert_transform(img_b_pose)[0:3, 3])
print("\n")
print(linear_distance)



######################## test##########################

a = np.array([[0.126901, -0.054710, 0.938],
                [0.076113, -0.057638, 0.942],
                [0.074728, -0.081546, 0.895],
                [0.125624, -0.081282, 0.893],
                [0.156072, -0.285685, 0.827],
                [0.019842, -0.280429, 0.851],
                [0.092248, -0.321462, 0.763],
                [-0.043618, -0.312796, 0.788]])


b = np.array([[0.46022323, 0.50710499, 0.28645349],
                [0.42473236, 0.47370705, 0.28595987],
                [0.38551146, 0.51143277, 0.28599533],
                [0.42059597, 0.54657292, 0.28665495],
                [0.34020177, 0.67224169, 0.13511288],
                [0.25803548, 0.56310284, 0.13381004],
                [0.24375798, 0.68313318, 0.13381931],
                [0.16232316, 0.57071841, 0.13304782]])

c = np.reshape(a[-2:], (2, 3))
test_a1 = np.reshape(c[0],(1,3))
test_a2 = np.reshape(c[1],(1,3))

c=np.reshape(b[-2:], (2, 3))
test_b1 = np.reshape(c[0],(1,3))
test_b2 = np.reshape(c[1],(1,3))

a = a[:-2]
b = b[:-2]
#r, t = rigid_transform_3D(a, b)
r,t = rigrid(a.reshape(3,a.shape[0]), b.reshape(-1,a.shape[0]))
print('r:',r)
print('t:',t)

bb = np.matmul(a, r) + t.reshape([1, 3])
print('b-bb:', b - bb)

c = np.matmul(test_a1, r) + t.reshape([1, 3])
print('c-test_b1:', c - test_b1)

c = np.matmul(test_a2, r) + t.reshape([1, 3])
print('c-test_b2:', c - test_b2)
