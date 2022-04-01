import os 
import sys
import time 
import numpy as np 
import open3d as o3d 
import copy 
import logging

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: % (message)s')
logger = logging.getLogger()


class ICP(object):
    
    def __init__(self, voxelSize = 0.004, source= None, initMethod = 'ransac'):
        
        self._voxelSize = voxelSize
        self._source = source.voxel_down_sample(0.001)
        self._source_down, self._source_fpfh =  self.preprocessPointCloud(
            source, voxelSize
        )
        self.initMethod = initMethod
        self.diameter = np.linalg.norm(np.asarray(source.get_max_bound())- np.asarray(source.get_min_bound()))
    
    def preprocessPointCloud(self, pcd, voxelSize):
        logger.debug(":: Downsample with a voxel size %.3f." % voxelSize)
        pcd_down = pcd.voxel_down_sample(voxelSize)
        
        radius_normal = voxelSize * 2
        logger.debug(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn=30)
        )
        #o3d.visualization.draw_geometries([pcd_down], point_show_normal=True)
        
        radius_feature = voxelSize * 7 
        logger.debug(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius= radius_feature, max_nn=120)
        )
        return pcd_down, pcd_fpfh
    
    def executeGlobalRegistration(self, target_down, target_fpfh):
        
        distance_threshold = self._voxelSize * 1.5 
        logger.debug(":: RANSAC registration on downsampled point clouds.")
        logger.debug(":: Since the downsampling voxel size is %.3f," %self._voxelSize)
        logger.debug(" we use a liberal distance threshold %.3f." % distance_threshold)
        
        if self.initMethod == 'fgr':
            regInit = o3d.pipelines.registration.registration_fast_based_feature_matching(
                self._source_down, target_down, self._source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance_threshold)
                )
        else:
            regInit = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                self._source_down, target_down, self._source_fpfh, target_fpfh, True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
                4, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 500)
            )
        return regInit
    
    def projectModel(self, model, camera, viz = False):
        radius = self.diameter * 100
        _, index = model.hidden_point_removal(camera, radius)
        modelProject = model.select_by_index(index)
        if viz:
            o3d.visualization.draw_geometries(
                [modelProject], window_name = "project", width=640, height=480)
        return modelProject, index

    def executeRegistration(self, target, init_trans= None, colorICP = False):
        target_down, target_fpfh = self.preprocessPointCloud(
            target, self._voxelSize
        )
        startTime = time.time()
        if init_trans is None:
            regInit = self.executeGlobalRegistration(target_down, target_fpfh)
            score = len(regInit.correspondence_set) / len(target_down.points)
            init_trans = regInit.transformation
            logger.info('RANSC time:{:.2f}s/image'.format(time.time()-startTime))
        else:
            threshold = 0.02
            evaluation = o3d.pipelines.registration.evaluate_registration(
                self._source_down, target_down, threshold, init_trans)
            score = evaluation
            
        startTime = time.time()
        target = target.voxel_down_sample(0.001)
        threshold = self._voxelSize * 1.0
        if colorICP:
            reg = o3d.pipelines.registration.registration_colored_icp(
                self._source, target,  threshold,  init_trans, 
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness = 1e-10,
                    relative_rmse = 1e-6,
                    max_iteration= 200))
        
        else:
            reg = o3d.pipelines.registration.registration_icp(
                self._source, target, threshold, init_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-10, relative_rmse=1e-7, 
                                                                  max_iteration=200))
        
        logger.info('executeICP time: {:.2f}s/image'.format(time.time()-startTime))
        return reg, score
    
    def drawRegistrationResult(self, target, transformation, windowName = 'result'):
        source_temp = copy.deepcopy(self._source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1 ,0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries(
            [source_temp, target_temp], window_name = windowName)
        
    def draw_inlier_outlier(self, cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)
        logger.info("Showing outliners(red) and inlier (grey):")
        outlier_cloud.paint_uniform_color([1, 0 ,0])
        inlier_cloud.paint_uniform_color([0.8 ,0.8 ,0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                          zoom= 0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up= [-0.0694, -0.9768, 0.2024])

if __name__ == "__main__":
    voxelSize = 0.004
    sourceFile = "/home/cyn/code/dense-visual-correspondence/modules/pose/model.pcd"
    source = o3d.io.read_point_cloud(sourceFile)
    icp = ICP(voxelSize=voxelSize, source=source)
    targetFile = "/home/cyn/code/dense-visual-correspondence/modules/pose/50000.pcd"
    target = o3d.io.read_point_cloud(targetFile)
    test_trans = np.array([[0.0, 0.0,  1.0,  1.0],
                                                  [1.0, 0.0,  0.0,  0.0],
                                                  [0.0, 1.0,  0.0,  0.0],
                                                  [0.0, 0.0,  0.0,  1.0]])
    target = target.transform(test_trans)
    reg, score = icp.executeRegistration(target)
    transformation = reg.transformation
    print(score)
    icp.drawRegistrationResult(target, transformation)
        
            
        

        
        