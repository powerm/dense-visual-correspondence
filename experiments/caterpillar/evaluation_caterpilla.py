import modules.utils.utils as utils
import sys
import logging
import os 
from dense_correspondence.evaluation.evaluation  import  DenseCorrespondenceEvaluation

utils.set_cuda_visible_devices([1]) # use this to manually set CUDA_VISIBLE_DEVICES
import os 

logging.basicConfig(level=logging.INFO)

EVALUATE = True
EVALUATE_CROSS_SCENE = False
logging_dir = "trained_models/caterpillar_new"
num_image_pairs = 500

descriptor_dim = [3, 6, 9]
M_background_list = [0.5, 1.0]
#network_list = [dict(model_class="Fuse", resnet_name="FuseNet"), dict(model_class="Resnet", resnet_name="Resnet34_8s")]

network_list = [dict(model_class="ResFuse", resnet_name="Resnet34_8s_atten_fuse"),dict(model_class="ResFuse", resnet_name="Resnet34_8s_fuse")]
# for  model  in network_list:
#     for M_background in M_background_list:
#         for d in descriptor_dim:
            
#             name = "caterpillar_%s_M_background_%.3f_%s_no_flip_new" %(model['resnet_name'],M_background, d)
#             model_folder = os.path.join(logging_dir, name)
#             model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder)
            
#             if EVALUATE:
#                 DCE = DenseCorrespondenceEvaluation
#                 DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs, 
#                                             cross_scene=EVALUATE_CROSS_SCENE)

folder_name = "caterpillar_new"
path_to_nets = os.path.join("/home/cyn/dataset/dense-net-entire/pdc/trained_models", folder_name)
all_nets = sorted(os.listdir(path_to_nets))


#name = 'caterpillar_FuseNet_M_background_0.500_3'
for  name in all_nets:
    model_folder = os.path.join(logging_dir, name)
    model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder)
    DCE = DenseCorrespondenceEvaluation
    DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs, 
                                                cross_scene=EVALUATE_CROSS_SCENE)


