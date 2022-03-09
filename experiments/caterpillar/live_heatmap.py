import sys
import os
import cv2
import numpy as np
import copy

import modules.utils.utils as utils
dc_source_dir = utils.getDenseCorrespondenceSourceDir()
sys.path.append(dc_source_dir)
sys.path.append(os.path.join(dc_source_dir, "dense_correspondence", "correspondence_tools"))
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, ImageType


import dense_correspondence
from dense_correspondence.evaluation.evaluation import *
from dense_correspondence.evaluation.plotting import normalize_descriptor
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork


import modules.utils.visualization as vis_utils

from modules.simple_pixel_correspondence_labeler.annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config, numpy_to_cv2


from  modules.user_interaction_heatmap_visualization.live_heatmap_visualization  import  HeatmapVisualization



COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0,255,0)


if __name__ == "__main__":
    

    utils.set_default_cuda_visible_devices()
    eval_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'caterpilla_evaluation.yaml')
    EVAL_CONFIG = utils.getDictFromYamlFilename(eval_config_filename)



    LOAD_SPECIFIC_DATASET = False
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()
    config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence', 'heatmap_vis', 'heatmap_caterpilla.yaml')
    config = utils.getDictFromYamlFilename(config_file)

    heatmap_vis = HeatmapVisualization(config, EVAL_CONFIG)
    print ("starting heatmap vis")
    heatmap_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()