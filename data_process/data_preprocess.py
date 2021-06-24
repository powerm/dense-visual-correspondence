import os
import sys
import numpy as np
import yaml

import   transformations
import  utils as myUtils




def poseFromMat(mat):
    '''
    Returns position, quaternion
    '''
    return np.array(mat[:3,3]), transformations.quaternion_from_matrix(mat, isprecise=True)


def dictFromMat(mat):
    pos, quat = poseFromMat(mat)
    pos = pos.tolist()
    quat = quat.tolist()
    d = dict()
    d['translation'] = dict()
    d['translation']['x'] = pos[0]
    d['translation']['y'] = pos[1]
    d['translation']['z'] = pos[2]

    d['quaternion'] = dict()
    d['quaternion']['w'] = quat[0]
    d['quaternion']['x'] = quat[1]
    d['quaternion']['y'] = quat[2]
    d['quaternion']['z'] = quat[3]

    return d

def sci(n):
      return "{:.8e}".format(n)

def saveToYaml(data, filename):
    with open(filename, 'w+') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def  npy_to_yaml(processed_dir):
    
    image_dir = os.path.join(processed_dir, 'images')
    posefile = os.path.join(image_dir, 'pose.npy')
    output_dir = image_dir
    
    poses = np.load(posefile)
    num = poses.shape[0]
    pose_data = dict()
    
    for idx in range(1,num+1):
        #pose = np.linalg.inv(poses[idx-1])
        print(idx)
        pose = poses[idx-1]
        pose_data[idx] = dict()
        d = pose_data[idx]
        transform_dict = dictFromMat(pose)
        d['camera_to_world'] = transform_dict
        d['timestamp'] = 1523132641382525436
        d['rgb_image_filename'] = "%06i_%s.png" % (idx, "rgb")
        d['depth_image_filename']= "%06i_%s.png" % (idx, "depth")

    saveToYaml(pose_data,  os.path.join(output_dir, 'pose_data.yaml'))
    
    

def npy_to_txt(processed_dir):
    
    image_dir = os.path.join(processed_dir, 'images')
    posefile = os.path.join(image_dir, 'pose.npy')
    output_dir = image_dir
    
    poses = np.load(posefile)
    num = poses.shape[0]
    pose_data = dict()
    
    for idx in range(1,num+1):
        #pose = np.linalg.inv(poses[idx-1])
        pose4 = poses[idx-1]
        pose_file_name =  "%06i_%s.txt" % (idx, "pose")
        pose_file_full_path = os.path.join(image_dir, pose_file_name)
        with open(pose_file_full_path, 'w') as the_file:
            the_file.write(" "+sci(pose4[0,0])+"     "+sci(pose4[0,1])+"     "+sci(pose4[0,2])+"     "+sci(pose4[0,3])+"    \n")
            the_file.write(" "+sci(pose4[1,0])+"     "+sci(pose4[1,1])+"     "+sci(pose4[1,2])+"     "+sci(pose4[1,3])+"    \n")
            the_file.write(" "+sci(pose4[2,0])+"     "+sci(pose4[2,1])+"     "+sci(pose4[2,2])+"     "+sci(pose4[2,3])+"    \n")
            the_file.write(" "+sci(pose4[3,0])+"     "+sci(pose4[3,1])+"     "+sci(pose4[3,2])+"     "+sci(pose4[3,3])+"    \n")


def  intrinsics_yaml_to_txt(processed_dir):
    image_dir = os.path.join(processed_dir, 'images')
    camera_info_yaml = os.path.join(image_dir, "camera_info.yaml")
    camera_info = myUtils.getDictFromYamlFilename(camera_info_yaml)
    K_matrix = camera_info['camera_matrix']['data']
    
    camera_intrinsics_out = os.path.join(image_dir, "camera-intrinsics.txt")
    with open(camera_intrinsics_out, 'w') as the_file:
        the_file.write(" "+sci(K_matrix[0])+"    "+sci(K_matrix[1])+"    "+sci(K_matrix[2])+"   \n")
        the_file.write(" "+sci(K_matrix[3])+"    "+sci(K_matrix[4])+"    "+sci(K_matrix[5])+"   \n")
        the_file.write(" "+sci(K_matrix[6])+"    "+sci(K_matrix[7])+"    "+sci(K_matrix[8])+"   \n")
    


def main(processed_dir):
    
    npy_to_yaml(processed_dir)
    npy_to_txt(processed_dir)
    intrinsics_yaml_to_txt(processed_dir)
    
    
    
    

if  __name__ == '__main__':
    

    processed_dir = "/home/cyn/dataset/dense-net-entire/pdc/logs_proto/000111_1/processed"

    main(processed_dir)


    
    
