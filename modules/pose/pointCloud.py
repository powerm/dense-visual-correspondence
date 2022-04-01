import numpy as np 
import  cv2 
import sys 
import os 
import open3d as o3d


def convertDepthToPointCloud(depth, cameraMatrix, distCoeffs=None, mask=None):
    """_summary_

    Args:
        depth (np.ndarray):  an array of shape (H, W)
        cameraMatrix (np.ndarray):  camera intrinsic
        distCoeffs (np.ndarray, optional): camera distortion coeffs. Defaults to None.
        mask (np.ndarray, optional): the target mask  an array of shape (H,W). Defaults to None.

    Returns:
        xyz (np.ndarray): the Pointcloud coordinate in meter
    """
    
    [height, width] = depth.shape 
    fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
    cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
    if distCoeffs is not None:
        depth = cv2.undistort(depth, cameraMatrix, distCoeffs)
    
    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.reshape(-1) -cx)/fx
    y = (v.reshape(-1)- cy)/fy
    
    z = depth.reshape(-1) /1000.0
    x = np.multiply(x, z)
    y = np.multiply(x, z)
    
    if mask:
        mask = mask.reshape(-1)
        x = x[mask]
        y = y[mask]
        z = z[mask]
    
    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]
    
    xyz = np.stack((x,y,z), axis=1)
    return xyz

def convertRGBDToPointCloud(color, depth, cameraMatrix, distCoeffs= None, mask=None):
    """_summary_

    Args:
        color (np.ndarray):  an image of shape (H, W, C) (in BGR order)
            This is the format used by OpenCV.
        depth (np.ndarray): an array of shape (H, W)
        cameraMatrix (np.ndarray):  camera intrinsic
        distCoeffs (np.ndarray, optional): camera distortion coeffs. Defaults to None.
        mask (np.ndarray, optional): the target mask  an array of shape (H,W). Defaults to None.

    Returns:
        xyz (np.ndarray): the Pointcloud coordinate in meter
        color (np.ndarray): the processed color sort as RGB
    """

    [height, width] = depth.shape 
    fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
    cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
    if distCoeffs is not None:
        depth = cv2.undistort(depth, cameraMatrix, distCoeffs)

    #color = color.reshape(-1, color.shape[-1][:, ::-1])
    color = color.reshape( height*width, -1)
    
    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.reshape(-1) -cx)/fx
    y = (v.reshape(-1)- cy)/fy
    
    z = depth.reshape(-1) /1000.0
    x = np.multiply(x, z)
    y = np.multiply(y, z)
    
    if mask is not None:
        mask = mask.reshape(-1)
        x = x[np.nonzero(mask)]
        y = y[np.nonzero(mask)]
        z = z[np.nonzero(mask)]
        
        color = color[np.nonzero(mask)]
    
    color = color[np.nonzero(z)]
    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]
    xyz = np.stack((x,y,z), axis=1)
    
    return xyz, color


def  getTargetPointCloud(colorImage, depthImage, cameraMatrixColor, distCoeffs=None,mask= None):
    xyz, color = convertRGBDToPointCloud(colorImage, depthImage, cameraMatrixColor, distCoeffs=distCoeffs, mask = mask)
    # filter = xyz[:,  2] <2
    # xyz = xyz[filter]
    # color = color[filter]
    targetPointCloud =  o3d.geometry.PointCloud()
    targetPointCloud.points = o3d.utility.Vector3dVector(xyz)
    targetPointCloud.colors = o3d.utility.Vector3dVector(color/256.0)
    
    return targetPointCloud








if __name__ == '__main__':
    import modules.utils.utils as utils
    from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
    config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'caterpillar_only.yaml')

    config = utils.getDictFromYamlFilename(config_filename)
    dataset = SpartanDataset(config=config)

    scene = "2018-04-10-16-02-59"
    img_a_index = dataset.get_random_image_index(scene)
    img_a_rgb, img_a_depth, img_a_mask, img_a_pose = dataset.get_rgbd_mask_pose(scene, img_a_index)

    img_b_index = dataset.get_img_idx_with_different_pose(scene, img_a_pose, num_attempts=50)
    img_b_rgb, img_b_depth, img_b_mask, img_b_pose = dataset.get_rgbd_mask_pose(scene, img_b_index)

    camera_intrinsics = dataset.get_camera_intrinsics(scene)
    camera_intrinsics_K = camera_intrinsics.K
    distCoeffs = np.array([
        0.05031626410741169, -0.2754641106388708, 0.0003722647488938247, 0.0003210956898043667, 0.2978803215933795])

    
    pcd_a = getTargetPointCloud(np.array(img_a_rgb),np.array(img_a_depth), camera_intrinsics_K, mask=np.array(img_a_mask))
    #pcd.transform([[1, 0 ,0, 0,],[0, -1 ,0, 0],[0, 0 ,-1, 0],[0, 0 ,0, 1]])
    pcd_a.transform(img_a_pose)
    #o3d.visualization.draw_geometries([pcd_a])
    
    pcd_b = getTargetPointCloud(np.array(img_b_rgb), np.array(img_b_depth), camera_intrinsics_K, mask = np.array(img_b_mask))
    o3d.visualization.draw_geometries([pcd_a, pcd_b])
    


    