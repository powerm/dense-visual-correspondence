import numpy as np 
import  sys 
import os
import  logging

import torch 
import open3d as o3d

from  modules.pose.rigrid import  rigid_transform_3D
from  modules.pose.transform3d import invert_transform, apply_transform_torch,get_point3d
import modules.utils.utils as utils
from  dense_correspondence.api.predictor import Predictor
from modules.pose.pointCloud import convertRGBDToPointCloud, getTargetPointCloud

from modules.pose.registration import ICP

utils.set_cuda_visible_devices([1])
logging.basicConfig(level=logging.INFO)



if __name__ == '__main__':
    eval_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'caterpilla_evaluation.yaml')
    EVAL_CONFIG = utils.getDictFromYamlFilename(eval_config_filename)

    predict = Predictor(EVAL_CONFIG)
    name =  'caterpillar_Resnet34_8s_M_background_0.500_3'
    predict.setup(name)
    # dcn = predict.load_network_from_config(name)
    dataset = predict.load_dataset_for_network(name)

    scene = "2018-04-10-16-02-59"
    img_a_index = dataset.get_random_image_index(scene)
    img_a_rgb, img_a_depth, img_a_mask, img_a_pose = dataset.get_rgbd_mask_pose(scene, img_a_index)
    img_b_index = dataset.get_img_idx_with_different_pose(scene, img_a_pose, num_attempts=50)
    img_b_rgb, img_b_depth, img_b_mask, img_b_pose = dataset.get_rgbd_mask_pose(scene, img_b_index)
    
    matches_a, matches_b, depth_a_vec, depth_b_vec = predict.batch_find_matches(scene, scene, img_a_index, img_b_index, num_matches=30)
    
    camera_intrinsics = dataset.get_camera_intrinsics(scene)
    camera_intrinsics_K = camera_intrinsics.K
    img_a_depth_numpy = np.asarray(img_a_depth)
    img_b_depth_numpy = np.asarray(img_b_depth)
    from dense_correspondence.correspondence_tools import correspondence_plotter
    #correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, matches_a, matches_b)
    
    aPoint_at_camera= get_point3d(matches_a, img_a_depth_numpy,camera_intrinsics_K)
    aPoint_at_world = apply_transform_torch(aPoint_at_camera, torch.from_numpy(img_a_pose).type(torch.FloatTensor))
    
    # b相对于世界坐标系的三维点
    bPoint_at_camera= get_point3d(matches_b, img_b_depth_numpy,camera_intrinsics_K)
    
    R, t = rigid_transform_3D(np.array(aPoint_at_world), np.array(bPoint_at_camera))
    trans = np.identity(4)
    trans[0:3, 0:3] = R
    trans[0:3, 3]= t.squeeze(1)
    
    print(trans)
    print("\n")
    print(invert_transform(img_b_pose))
    
    pcd_a = getTargetPointCloud(np.array(img_a_rgb),np.array(img_a_depth), camera_intrinsics_K, mask=np.array(img_a_mask))
    #pcd.transform([[1, 0 ,0, 0,],[0, -1 ,0, 0],[0, 0 ,-1, 0],[0, 0 ,0, 1]])
    pcd_a.transform(img_a_pose)
    #pcd_a.transform(trans)
    
    pcd_b = getTargetPointCloud(np.array(img_b_rgb), np.array(img_b_depth), camera_intrinsics_K, mask = np.array(img_b_mask))
    #o3d.visualization.draw_geometries([pcd_a, pcd_b])
    
    voxelSize = 0.004
    
    icp = ICP(voxelSize=voxelSize, source=pcd_a)
    reg, score = icp.executeRegistration(pcd_b, init_trans= trans)
    transformation= reg.transformation
    print(reg)
    print("\n")
    print(score)
    icp.drawRegistrationResult(pcd_b, transformation)
    
    
    