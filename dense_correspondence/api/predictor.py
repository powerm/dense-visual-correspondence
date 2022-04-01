

import os 

import  logging 
import  numpy as np 

import torch 

from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
import modules.utils.visualization as vis_utils
from dense_correspondence.correspondence_tools.correspondence_finder import random_sample_from_masked_image, \
    random_sample_from_masked_image_torch

import modules.utils.utils as utils




class  Predictor(object):
    
    def __init__(self, config):
        self._config = config
        self._dataset = None
        self._dcn = None
    
    def setup(self, name):
        dcn = self.load_network_from_config(name)
        dataset = self.load_dataset_for_network(name)
        
        
    
    @property
    def config(self):
        return self._configs
    
    def load_network_from_config(self, name):
        """
        Loads a network from config file. Puts it in eval mode by default
        :param name:
        :type name:
        :return: DenseCorrespondenceNetwork
        :rtype:
        """
        if name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" %(name))


        path_to_network_params = self._config["networks"][name]["path_to_network_params"]
        path_to_network_params = utils.convert_data_relative_path_to_absolute_path(path_to_network_params, assert_path_exists=True)
        model_folder = os.path.dirname(path_to_network_params)

        dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder, model_param_file=path_to_network_params)
        dcn.eval()
        self.dcn = dcn
        return dcn

    def load_dataset_for_network(self, network_name):
        """
        Loads a dataset for the network specified in the config file
        :param network_name: string
        :type network_name:
        :return: SpartanDataset
        :rtype:
        """
        if network_name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" %(network_name))

        network_folder = os.path.dirname(self._config["networks"][network_name]["path_to_network_params"])
        network_folder = utils.convert_data_relative_path_to_absolute_path(network_folder, assert_path_exists=True)
        dataset_config = utils.getDictFromYamlFilename(os.path.join(network_folder, "dataset.yaml"))

        dataset = SpartanDataset(config_expanded=dataset_config)
        self.dataset = dataset
        return dataset
    
    def load_dataset(self):
        """
        Loads a SpartanDatasetMasked object
        For now we use a default one
        :return:
        :rtype: SpartanDatasetMasked
        """

        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'dataset',
                                   'spartan_dataset_masked.yaml')

        config = utils.getDictFromYamlFilename(config_file)

        dataset = SpartanDataset(mode="test", config=config)

        return dataset
    
    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self.load_dataset()
        return self._dataset
    
    @dataset.setter
    def dataset(self, value):
        self._dataset = value
    
    @property
    def dcn(self):
        return self._dcn
    
    @dcn.setter
    def dcn(self, value):
        self._dcn = value
        
    
    def batch_find_matches(self, scene_name_a, scene_name_b, img_a_idx,  img_b_idx, num_matches= 20, dataset=None, dcn=None):
        
        #dataset = self.dataset
        if dataset is None:
            dataset = self.dataset
        rgb_a, depth_a, mask_a, pose_a   = dataset.get_rgbd_mask_pose(scene_name_a, img_a_idx)
        rgb_b, depth_b, mask_b, pose_b = dataset.get_rgbd_mask_pose(scene_name_b, img_b_idx)
        
        depth_a = np.asarray(depth_a)
        depth_b = np.asarray(depth_b)
        
        if dcn is None:
            dcn = self.dcn
        
        res_a = dcn.forward_single_image_tensor(rgb_a, depth_a)
        res_b = dcn.forward_single_image_tensor(rgb_b, depth_b)
        
        camera_intrinsics_a = dataset.get_camera_intrinsics(scene_name_a)
        camera_intrinsics_b = dataset.get_camera_intrinsics(scene_name_b)
        if not np.allclose(camera_intrinsics_a.K, camera_intrinsics_b.K):
            print ("Currently cannot handle two different camera K matrices in different scenes!")
            print ("But you could add this...")
        camera_intrinsics_matrix = camera_intrinsics_a.K
        
        return self.compute_descriptor_match(depth_a, depth_b, mask_a, mask_b,
                                      res_a, res_b, num_matches= num_matches)
        
    
    def compute_descriptor_match(self, depth_a, depth_b, mask_a, mask_b, 
                                 res_a, res_b, rgb_a = None, rgb_b=None, num_matches= 20):
        
        assert (depth_a.shape == depth_b.shape)
        image_width  = depth_a.shape[1]
        image_height = depth_a.shape[0]
        
        mask_a = np.asarray(mask_a)
        uv_a_vec = random_sample_from_masked_image_torch(mask_a,  num_matches)
        if uv_a_vec[0] is None:
            return (None, None)
        uv_a_vec_flattened = uv_a_vec[1]*image_width + uv_a_vec[0]
        depth_a_torch = torch.from_numpy(depth_a.copy()).type(torch.FloatTensor)
        depth_a_torch  = torch.squeeze(depth_a_torch, 0)
        depth_a_torch = depth_a_torch.view(-1,1)
        DEPTH_IM_SCALE = 1000.0
        depth_a_vec = torch.index_select(depth_a_torch, 0, uv_a_vec_flattened)*1.0/DEPTH_IM_SCALE
        depth_a_vec = depth_a_vec.squeeze(1)
        # Prune based on
        # Case 1: depth is zero (for this data, this means no-return)
        nonzero_indices = torch.nonzero(depth_a_vec)
        nonzero_indices = nonzero_indices.squeeze(1)
        depth_vec = torch.index_select(depth_a_vec, 0, nonzero_indices)
        u_a_pruned = torch.index_select(uv_a_vec[0], 0, nonzero_indices)
        v_a_pruned = torch.index_select(uv_a_vec[1], 0, nonzero_indices)
        uv_a_pruned_flattened = v_a_pruned*image_width + u_a_pruned
        
        matches_b_pre, best_match_diff, best_match_heatmap=DenseCorrespondenceNetwork.find_batch_best_matches(res_a, res_b, uv_a_pruned_flattened)
        
        depth_b_torch = torch.from_numpy(depth_b.copy()).type(torch.FloatTensor)
        depth_b_torch = torch.squeeze(depth_b_torch, 0)
        depth_b_torch = depth_b_torch.view(-1,1)
        DEPTH_IM_SCALE = 1000.0
        depth_b_vec = torch.index_select(depth_b_torch, 0, matches_b_pre.data.cpu())*1.0/DEPTH_IM_SCALE
        depth_b_vec = depth_b_vec.squeeze(1)
        # Prune based on
        # Case 1: depth is zero (for this data, this means no-return)
        nonzero_indices = torch.nonzero(depth_b_vec)
        nonzero_indices = nonzero_indices.squeeze(1)
        depth_b_vec = torch.index_select(depth_b_vec, 0, nonzero_indices)
        
        matches_b_pre = matches_b_pre.data.cpu()
        uv_b_vec = ((matches_b_pre%image_width), (matches_b_pre//image_width))
        
        u_b_pruned = torch.index_select(uv_b_vec[0], 0, nonzero_indices)
        v_b_pruned = torch.index_select(uv_b_vec[1], 0, nonzero_indices)
        
        u_a_pruned = torch.index_select(u_a_pruned, 0, nonzero_indices)
        v_a_pruned = torch.index_select(v_a_pruned, 0, nonzero_indices)
        depth_a_vec = torch.index_select(depth_a_vec, 0, nonzero_indices)
       
        
        return (u_a_pruned, v_a_pruned), (u_b_pruned, v_b_pruned), depth_a_vec.numpy(), depth_b_vec.numpy()
        
        
        


if __name__ == '__main__':
    
    from  modules.pose.rigrid import  rigid_transform_3D
    from dense_correspondence.correspondence_tools import correspondence_finder, correspondence_plotter
    from  modules.pose.transform3d import invert_transform, apply_transform_torch,get_point3d
    utils.set_cuda_visible_devices([1])
    logging.basicConfig(level=logging.INFO)
    
    eval_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'caterpilla_evaluation.yaml')
    EVAL_CONFIG = utils.getDictFromYamlFilename(eval_config_filename)
    
    predict = Predictor(EVAL_CONFIG)
    name =  'caterpillar_Resnet34_8s_M_background_0.500_3'
    predict.setup(name)
    # dcn = predict.load_network_from_config(name)
    dataset = predict.load_dataset_for_network(name)
    
    scene = "2018-04-10-16-02-59"
    img_a_index = dataset.get_random_image_index(scene)
    img_a_rgb, img_a_depth, img_a_mask, img_a_pose = dataset.get_rgbd_mask_pose(scene, img_a_index)

    img_b_index = dataset.get_img_idx_with_different_pose(scene, img_a_pose, num_attempts=50)
    img_b_rgb, img_b_depth, img_b_mask, img_b_pose = dataset.get_rgbd_mask_pose(scene, img_b_index)
        
    matches_a, matches_b, depth_a_vec, depth_b_vec = predict.batch_find_matches(scene,scene,img_a_index, img_b_index, num_matches= 30)
    
    camera_intrinsics = dataset.get_camera_intrinsics(scene)
    camera_intrinsics_K = camera_intrinsics.K
    img_a_depth_numpy = np.asarray(img_a_depth)
    img_b_depth_numpy = np.asarray(img_b_depth)
    
    correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, matches_a, matches_b)
    
    aPoint_at_camera= get_point3d(matches_a, img_a_depth_numpy,camera_intrinsics_K)
    aPoint_at_world = apply_transform_torch(aPoint_at_camera, torch.from_numpy(img_a_pose).type(torch.FloatTensor))
    
    # b相对于世界坐标系的三维点
    bPoint_at_camera= get_point3d(matches_b, img_b_depth_numpy,camera_intrinsics_K)
    
    R, t = rigid_transform_3D(np.array(aPoint_at_world), np.array(bPoint_at_camera))
    
    def  mse(B, B2):
        err = B2 - B 
        err = err*err
        err = np.sum(err)
        rmse = np.sqrt(err/B.shape[1])
        return rmse
    
    print(R)
    print("\n")
    print(t)
    print("\n")
    print(invert_transform(img_b_pose))
    
    point3d_b_pre =  R@(np.array(aPoint_at_world)) + t
    print("mse:", mse(np.array(bPoint_at_camera), point3d_b_pre))
    print("\n")