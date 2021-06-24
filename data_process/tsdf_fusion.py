import open3d as o3d
import numpy as np
import os
import shutil
import yaml
import utils as myUtils




def format_data_for_tsdf(image_folder):
    """
    Processes the data into the format needed for tsdf-fusion algorithm
    """

    # image_folder = os.path.join(data_folder, 'images')
    camera_info_yaml = os.path.join(image_folder, "camera_info.yaml")


    camera_info = myUtils.getDictFromYamlFilename(camera_info_yaml)

    K_matrix = camera_info['camera_matrix']['data']
    # print K_matrix
    n = K_matrix[0]

    def sci(n):
      return "{:.8e}".format(n)

    camera_intrinsics_out = os.path.join(image_folder,"camera-intrinsics.txt")
    with open(camera_intrinsics_out, 'w') as the_file:
        the_file.write(" "+sci(K_matrix[0])+"    "+sci(K_matrix[1])+"    "+sci(K_matrix[2])+"   \n")
        the_file.write(" "+sci(K_matrix[3])+"    "+sci(K_matrix[4])+"    "+sci(K_matrix[5])+"   \n")
        the_file.write(" "+sci(K_matrix[6])+"    "+sci(K_matrix[7])+"    "+sci(K_matrix[8])+"   \n")


    ### HANDLE POSES

    pose_data_yaml = os.path.join(image_folder, "pose_data.yaml")
    with open(pose_data_yaml, 'r') as stream:
        try:
            pose_data_dict = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print(pose_data_dict[1])

    for i in pose_data_dict:
        # print i
        # print pose_data_dict[i]
        pose4 = myUtils.homogenous_transform_from_dict(pose_data_dict[i]['camera_to_world'])
        depth_image_filename = pose_data_dict[i]['depth_image_filename']
        prefix = depth_image_filename.split("depth")[0]
        print(prefix)
        pose_file_name = prefix+"pose.txt"
        pose_file_full_path = os.path.join(image_folder, pose_file_name)
        with open(pose_file_full_path, 'w') as the_file:
            the_file.write(" "+sci(pose4[0,0])+"     "+sci(pose4[0,1])+"     "+sci(pose4[0,2])+"     "+sci(pose4[0,3])+"    \n")
            the_file.write(" "+sci(pose4[1,0])+"     "+sci(pose4[1,1])+"     "+sci(pose4[1,2])+"     "+sci(pose4[1,3])+"    \n")
            the_file.write(" "+sci(pose4[2,0])+"     "+sci(pose4[2,1])+"     "+sci(pose4[2,2])+"     "+sci(pose4[2,3])+"    \n")
            the_file.write(" "+sci(pose4[3,0])+"     "+sci(pose4[3,1])+"     "+sci(pose4[3,2])+"     "+sci(pose4[3,3])+"    \n")




def main(processed_dir):
    
    if os.path.exists(processed_dir):
        images_dir =  os.path.join(processed_dir, 'images')
    else:
        print("There is not exist the  dictory!")
        return 0
    
    voxel_size =  0.0025     # 2.5mm
    trunc = np.inf
    # Intrinsics:
    width   = 640
    height  = 480
    fx      = 1035.4066400628542
    fy      = 1164.782443530703
    cx      = 318.33921710833334
    cy      = 237.3960612108707

    ##   camera intrinsic  param
    cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    ##  The number of camera pose
    posefile = os.path.join(images_dir, 'pose.npy')
    poses = np.load(posefile)
    num_of_poses = poses.shape[0]
    
    ##  init TSDF volume, use NoColor type
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length = 0.002,  # meters # ~ 1cm
    sdf_trunc   =  0.002*5,  # meters # ~ several voxel_lengths
    color_type  =  o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

    #  zero color image
    color_zero_arr = np.zeros((height, width))
    color_zero = o3d.geometry.Image((color_zero_arr).astype(np.float32))
    
    for i in range(num_of_poses):
        if os.path.exists(os.path.join(images_dir, "%06d_depth.png"%(i))) is not True:
            print(os.path.exists(os.path.join(images_dir, "%06d_depth.png"%(i))))
            continue
        print("Integrate %s-th image into the volume."%i)
        depth = o3d.io.read_image(os.path.join(images_dir, "%06d_depth.png"%(i)))
        depth = np.asarray(depth).astype(np.float32)
        #  devided 10 because depth  storage in 0.1 mm , covert to 1mm
        depth = o3d.geometry.Image(depth/10)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_zero, # !
        depth=depth, 
        depth_trunc=  1.5, # truncate the depth at z  = 1.5m
        convert_rgb_to_intensity=True)
        cam_pose = np.loadtxt(os.path.join(images_dir, "%06d_pose.txt"%(i)))
        
        #volume.integrate(rgbd, cameraIntrinsics, np.linalg.inv(cam_pose))
        volume.integrate(rgbd, cameraIntrinsics, cam_pose)
        print("integrate sucess!")
    
    mesh = volume.extract_triangle_mesh()
    #print(mesh.compute_vertex_normals())
    #mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # flip!
    o3d.io.write_triangle_mesh(os.path.join(processed_dir, 'fusion_mesh.ply') , mesh)
    o3d.visualization.draw_geometries([mesh])
    


if __name__ == '__main__':
    # processed_dir="/home/cyn/dataset/pdc/logs_test/2021-04-16-11-29-34/processed"
    # images_dir = "/home/cyn/dataset/pdc/logs_test/2021-04-16-11-29-34/processed/images"

    #processed_dir="/home/cyn/dataset/pdc/logs_test/000076_high_1"
    #images_dir = "/home/cyn/dataset/pdc/logs_test/000076_high_1/images"
    #processed_dir = "/home/cyn/dataset/dense-net-entire/pdc/logs_proto/000076_high_1"
    #images_dir = "/home/cyn/dataset/dense-net-entire/pdc/logs_proto/000076_high_1/images"

    processed_dir="/home/cyn/dataset/dense-net-entire/pdc/logs_proto/000111_1/processed"
    main(processed_dir)
