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




train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'training', 'training.yaml')
train_config = utils.getDictFromYamlFilename(train_config_file)
TRAIN = True
EVALUATE = True
EVALUATE_CROSS_SCENE = False
network_list = [dict(model_class="Resnet", resnet_name="Resnet34_8s"), dict(model_class="ResFuse", resnet_name="Resnet34_8s_atten_fuse"), \
    dict(model_class="Fuse", resnet_name="FuseNet"), dict(model_class="ResFuse", resnet_name="Resnet34_8s_fuse"),\
        dict(model_class="ResFuse", resnet_name="Resnet34_8s_cat_fuse")]
num_iterations = 3500
num_image_pairs = 500


batch_config_file = config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'experiments', 'batch_train', 'batch_train_all_1.yaml')
batch_config  = utils.getDictFromYamlFilename(batch_config_file)

for  a_config  in  batch_config:
    config_filename = a_config['config_file']
    logging_dir = a_config['logging_dir']
    descriptor_dim = a_config['descriptor_dim']
    M_background_list = a_config['M_background_list']
    
    config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                                'dataset', 'composite', config_filename)
    config = utils.getDictFromYamlFilename(config_filename)
    dataset = SpartanDataset(config=config)

    for model in network_list:
        for M_background in M_background_list:
            for d in descriptor_dim:
                print("d:", d)
                print("M_background:", M_background)
                print("training descriptor of dimension %d" %(d))
                print("network is " , model['resnet_name'])
                
                train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
                train_config = utils.getDictFromYamlFilename(train_config_file)
                name = "caterpillar_%s_M_background_%.3f_%s" %(model['resnet_name'],M_background, d)

                train._config["training"]["logging_dir"] = logging_dir
                train._config["training"]["logging_dir_name"] = name
                train._config["dense_correspondence_network"]["descriptor_dimension"] = d
                train._config['dense_correspondence_network']['backbone'] = model
                train._config["loss_function"]["M_background"] = M_background
                if model['model_class'] == "Fuse" or model['model_class'] == "ResFuse":
                    dataset._trans = False
                else:
                    dataset._trans = True
                
                if TRAIN:
                    train.run()
                print("finished training descriptor of dimension %d" %(d))

                
                model_folder = os.path.join(logging_dir, name)
                model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder)
                
                if EVALUATE:
                    DCE = DenseCorrespondenceEvaluation
                    DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs, 
                                                cross_scene=EVALUATE_CROSS_SCENE)      
        