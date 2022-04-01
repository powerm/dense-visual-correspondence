import modules.utils.utils as utils
#utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
import sys
import logging

# utils.set_default_cuda_visible_devices()
utils.set_cuda_visible_devices([1]) # use this to manually set CUDA_VISIBLE_DEVICES

from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
logging.basicConfig(level=logging.INFO)

isolated_dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', "caterpillar_baymax_starbot_all_front_single_only.yaml")

cluttered_dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', "caterpillar_baymax_starbot_all_front.yaml")

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'training', 'training.yaml')

network_list = [dict(model_class="Resnet", resnet_name="Resnet34_8s"), dict(model_class="ResFuse", resnet_name="Resnet34_8s_atten_fuse"), \
    dict(model_class="Fuse", resnet_name="FuseNet"), dict(model_class="ResFuse", resnet_name="Resnet34_8s_fuse"),\
        dict(model_class="ResFuse", resnet_name="Resnet34_8s_cat_fuse")]

logging_dir = "trained_models/result/multi_object/cluttered_scene"
num_iterations = 5000
num_image_pairs = 300
debug = False

TRAIN = False
EVALUATE = True

# num_image_pairs = 10
# num_iterations = 10
d_list = [9,16,32]
M_background_list = [0.5, 1.0, 1.5, 2.0]

network_dict = dict()

for model in network_list:
    for d in d_list:
        for M_background in M_background_list:
            # load dataset and training config
            dataset_config = utils.getDictFromYamlFilename(isolated_dataset_config_filename)
            dataset = SpartanDataset(config=dataset_config)
            train_config = utils.getDictFromYamlFilename(train_config_file)

            name = "multi_object_isolated_%s_M_background_%.1f_%d" %(model['resnet_name'], M_background, d)
            print("training %s" %(name))
            train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
            train._config["training"]["logging_dir"] = logging_dir
            train._config["training"]["logging_dir_name"] = name
            train._config["training"]["num_iterations"] = num_iterations
            train._config["dense_correspondence_network"]["descriptor_dimension"] = d
            train._config['dense_correspondence_network']['backbone'] = model
            train._config["loss_function"]["M_background"] = M_background
            train._config["training"]["data_type_probabilities"]["SINGLE_OBJECT_WITHIN_SCENE"] = 0.5
            train._config["training"]["data_type_probabilities"]["DIFFERENT_OBJECT"] = 0.5
            if model['model_class'] == "Fuse" or model['model_class'] == "ResFuse":
                dataset._trans = False
            else:
                dataset._trans = True


            if TRAIN:
                train.run()
            print("finished training descriptor of dimension %d" %(d))
            del train

            # now do evaluation
            print("running evaluation on network %s" %(name))
            model_folder = os.path.join(logging_dir, name)
            #model_folder = utils.convert_to_absolute_path(model_folder)
            network_dict[name] = model_folder
            
            if EVALUATE:
                DCE = DenseCorrespondenceEvaluation
                isolated_dataset_config = utils.getDictFromYamlFilename(isolated_dataset_config_filename)
                dataset = SpartanDataset(config=isolated_dataset_config)
                DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs, 
                                            save_folder_name="analysis_isolated_scene", dataset=dataset)
                
                cluttered_dataset_config = utils.getDictFromYamlFilename(cluttered_dataset_config_filename)
                cluttered_dataset = SpartanDataset(config=cluttered_dataset_config)
                DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs, 
                                            save_folder_name="analysis_cluttered_scene",
                                            dataset=cluttered_dataset)
                
            print("finished running evaluation on network %s" %(name))
            
            # also evaluate them on cross-scene data