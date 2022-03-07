import numpy as np
import shutil
import os
import glob



def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")


def  process_image(input_dir):
    processed_path = os.path.join(input_dir, "processed")
    images_path = os.path.join(input_dir, 'processed', 'images')
    mkdir(processed_path)
    mkdir(images_path)
    
    pose_origin = os.path.join(input_dir, 'pose.npy')
    shutil.copy(pose_origin, os.path.join(images_path, 'pose.npy'))
    shutil.copy(os.path.join(input_dir, 'camera_info.yaml'), os.path.join(images_path, 'camera_info.yaml'))

    depth_origin = os.path.join(input_dir, 'depth')
    rgb_origin = os.path.join(input_dir,'rgb')
    file_num_list = glob.glob(os.path.join(depth_origin,'*.png'))
    file_num = len(file_num_list)
    for i in range(file_num):
        shutil.copy(os.path.join(depth_origin, "%06i.png" % (i+1)),  os.path.join(images_path, "%06i_%s.png" % (i+1, "depth")))
        shutil.copy(os.path.join(rgb_origin, "%06i.png" % (i+1)),  os.path.join(images_path, "%06i_%s.png" % (i+1, "rgb")))

def main(input_dir):
    process_image(input_dir)


if __name__ == '__main__':
    pass