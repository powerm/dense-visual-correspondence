import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch

# utils.add_dense_correspondence_to_python_path()

import dense_correspondence
import modules.utils.utils as utils
from dense_correspondence.evaluation.evaluation import *
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
from dense_correspondence.dataset.dense_correspondence_dataset_masked import ImageType

from PIL import Image
from plotting import normalize_descriptor
import time

def make_descriptors_images( dcn, log_folder, save_images_dir, make_masked_video=False):
    image_folder = rgb_filename = os.path.join(log_folder, "processed", "images")
    for img_file in sorted(os.listdir(image_folder)):
        #print i
        start = time.time()
        if "rgb.png" not in img_file:
            continue
        
        idx_str = img_file.split("_rgb")[0]
        img_file_fullpath = os.path.join(image_folder, img_file)
        depth_file = idx_str + '_depth.png'
        depth_file_fullpath = os.path.join(image_folder, depth_file)
        rgb_a = Image.open(img_file_fullpath).convert('RGB')
        depth_a = Image.open(depth_file_fullpath)

        # compute dense descriptors
        # This takes in a PIL image!
        #rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = dcn.forward_single_image_tensor(rgb_a,  depth_a).data.cpu().numpy()
        res_a = normalize_descriptor(res_a, descriptor_image_stats["mask_image"])

        # This chunk of code would produce masked descriptors
        # MASK
        if make_masked_video:
            mask_name = idx_str + "_mask.png"
            mask_filename = os.path.join(log_folder, "processed", "image_masks", mask_name)
            mask = np.asarray(Image.open(mask_filename))
            mask_three_channel = np.zeros((480,640,3))
            for j in range(3):
                mask_three_channel[:,:,j] = mask
            res_a_masked = res_a * mask_three_channel

        
        # save rgb image, descriptor image, masked descriptor image
        
        save_file_name = os.path.join(save_images_dir, idx_str + "_res.png")
        plt.imsave(save_file_name, res_a)
        print("forward and saving at rate", time.time()-start)
        
        

def make_videos(log_folder, save_images_dir, make_masked_video=False):
    # make an rgb only dir
    log_name = os.path.basename(log_folder)
    print("log_name", log_name)
    processed_folder = os.path.join(log_folder, 'processed')
    videos_folder = os.path.join(processed_folder, 'videos2')
    if not os.path.isdir(videos_folder):
        os.makedirs(videos_folder)
    
#     rgb_only_path = os.path.join(os.path.dirname(full_rgb_only_path),"rgb_only")
#     os.system("mkdir -p "+ rgb_only_path)
#     os.system("cp "+full_rgb_only_path+"/*rgb.png "+ rgb_only_path)
    
    # make descriptor video
    print("making descriptor video")
    video_des = log_name + "_video_descriptors.mp4"
    video_des_full_filename = os.path.join(videos_folder, video_des)
    os.chdir(save_images_dir)
    cmd = "ffmpeg -framerate 30 -pattern_type glob -i '*res.png' -c:v libx264 -r 30 "\
              + video_des_full_filename
        
    print( "descriptor video command:\n", cmd)
    os.system(cmd)
    print ("done making descriptors")
    
    # make rgb video
    # save it in log_folder/processed/videos
    print("making rgb video")
    os.chdir(save_images_dir)
    video_rgb = log_name + "_video_rgb.mp4"
    video_rgb_full_filename = os.path.join(videos_folder, video_rgb)
    
    rgb_images_folder = os.path.join(processed_folder, 'images')
    print("rgb_images_folder", rgb_images_folder)
    os.chdir(rgb_images_folder)
    cmd = "ffmpeg -framerate 30 -pattern_type glob -i '*rgb.png' -c:v libx264 -r 30 " + video_rgb_full_filename
        
    print("rgb video command:\n", cmd)
    os.system(cmd)
    
    
    # make rgb video mac friendly
    os.chdir(videos_folder)
    cmd = "ffmpeg \
      -i "+ video_rgb + " -pix_fmt yuv420p " + video_rgb.split(".mp4")[0]+"_mac.mp4"
    print(cmd)
    os.system(cmd)
    
    # make descriptor video mac friendly
    os.chdir(videos_folder)
    cmd = "ffmpeg \
      -i "+ video_des + " -pix_fmt yuv420p "+ video_des.split(".mp4")[0] + "_mac.mp4"
    os.system(cmd)
    
    
    # merge the videos!
    os.chdir(videos_folder)
    cmd = "ffmpeg \
      -i "+ video_rgb +" \
      -i "+ video_des +"\
      -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
      -map [vid] \
      -c:v libx264 \
      -crf 23 \
      -preset veryfast \
      output_" + log_name + ".mp4"

    print(cmd)
    os.system(cmd)
    


if __name__ == "__main__":
  config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 
                                'dense_correspondence', 'evaluation', 'caterpilla_evaluation.yaml')
  config = utils.getDictFromYamlFilename(config_filename)
  default_config = utils.get_defaults_config()

  utils.set_cuda_visible_devices([0])
  dce = DenseCorrespondenceEvaluation(config)
  DCE = DenseCorrespondenceEvaluation
  log_list = []
  # mugs
  #network_name = "caterpillar_Resnet34_8s_atten_fuse_M_background_0.500_3_no_flip_new"\
  network_name = "caterpillar_Resnet34_8s_M_background_0.500_3"
  # log_list.append("2018-05-18-16-26-26") # many mugs, moving robot
  log_list.append("2018-04-16-14-25-19") # may mugs, stationary robot
  dcn = dce.load_network_from_config(network_name)
  dcn.eval()
  dataset = dcn.load_training_dataset()

  descriptor_image_stats = dcn.descriptor_image_stats

  logs_special_prefix = "/home/cyn/dataset/dense-net-entire/pdc/logs_proto"
  make_masked_video = False

  for log in log_list:
    log_folder = os.path.join(logs_special_prefix, log)
    save_images_dir = os.path.join(log_folder, "processed", "video_images_2")
    if not os.path.isdir(save_images_dir):
      os.makedirs(save_images_dir)
    print("save_images_dir", save_images_dir)
      
    make_descriptors_images(dcn,log_folder, save_images_dir, make_masked_video=make_masked_video)
    make_videos(log_folder, save_images_dir, make_masked_video=make_masked_video)

