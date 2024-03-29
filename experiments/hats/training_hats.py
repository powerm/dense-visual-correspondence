# Hats
# Train hats, both class consistent and class discriminatory

import modules.utils.utils as utils
#utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
import sys
import logging

# utils.set_default_cuda_visible_devices()
utils.set_cuda_visible_devices([0]) # use this to manually set CUDA_VISIBLE_DEVICES

from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
logging.basicConfig(level=logging.INFO)

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'hat_train_6.yaml')
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'training', 'training.yaml')

train_config = utils.getDictFromYamlFilename(train_config_file)

dataset = SpartanDataset(config=config)
logging_dir = "trained_models/hats_6"
num_iterations = 3500
num_image_pairs = 100

TRAIN = False
EVALUATE = True

#  Class Consistent

descriptor_dim = [3,6,9]
M_background_list = [0.5, 2.0]

for M_background in M_background_list:
    for d in descriptor_dim:
        print("\n")
        print("d:", d)
        print("M_background:", M_background)
        print("training descriptor of dimension %d" %(d))
        
        train_config = utils.getDictFromYamlFilename(train_config_file)
        train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
        name = "hats_consistent_M_background_%.3f_%s" %(M_background, d)

        train._config["training"]["logging_dir"] = logging_dir
        train._config["training"]["logging_dir_name"] = name
        train._config["dense_correspondence_network"]["descriptor_dimension"] = d
        train._config["loss_function"]["M_background"] = M_background

        if TRAIN:
            train.run()
        print("finished training descriptor of dimension %d" %(d))

         
        model_folder = os.path.join(logging_dir, name)
        model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder)
        
        if EVALUATE:
            DCE = DenseCorrespondenceEvaluation
            DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs, cross_scene=False)      
    
