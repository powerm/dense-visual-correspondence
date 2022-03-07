import os
import data_preprocess
import tsdf_fusion
import mesh_crop
import crop_detection_render
import  image_process

def main(processed_dir):
    data_preprocess.main(processed_dir)
    tsdf_fusion.main(processed_dir)
    mesh_crop.main(processed_dir)
    crop_detection_render.main(processed_dir)

if __name__ == "__main__":
    
    base_dir = '/home/cyn/dataset/dense-net-entire/pdc/logs_proto'
    input_dirs = []
    
    for i in range(8):
        str = "000112_%i"% (i+1)
        input = os.path.join(base_dir, str)
        input_dirs.append(input)
    
    for  n,  input_dir in  enumerate(input_dirs):
        image_process.main(input_dir)
        main(os.path.join(input_dir, 'processed'))
    
    #processed_dir = "/home/cyn/dataset/dense-net-entire/pdc/logs_proto/000111_5/processed"
    #main(processed_dir)
