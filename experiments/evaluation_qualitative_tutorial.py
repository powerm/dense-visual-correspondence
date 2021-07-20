import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import modules.utils.utils as utils
#utils.add_dense_correspondence_to_python_path()

import dense_correspondence
from dense_correspondence.evaluation.evaluation import *
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
from dense_correspondence.dataset.dense_correspondence_dataset_masked import ImageType


config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 
                               'dense_correspondence', 'evaluation', 'evaluation.yaml')
config = utils.getDictFromYamlFilename(config_filename)
default_config = utils.get_defaults_config()

# utils.set_cuda_visible_devices([0])
dce = DenseCorrespondenceEvaluation(config)
DCE = DenseCorrespondenceEvaluation

network_name = "caterpillar_3"
dcn = dce.load_network_from_config(network_name)
dataset = dcn.load_training_dataset()
DenseCorrespondenceEvaluation.evaluate_network_qualitative(dcn, dataset=dataset, randomize=True)