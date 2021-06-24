import os
import data_preprocess
import tsdf_fusion
import mesh_crop
import crop_detection_render

def main(processed_dir):
    data_preprocess.main(processed_dir)
    tsdf_fusion.main(processed_dir)
    mesh_crop.main(processed_dir)
    crop_detection_render.main(processed_dir)

if __name__ == "__main__":
    processed_dir = "/home/cyn/dataset/dense-net-entire/pdc/logs_proto/000111_5/processed"
    main(processed_dir)
