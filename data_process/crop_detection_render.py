from __future__ import absolute_import, print_function
import math
import os
import  matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.lib import utils 
import open3d
import cv2 as cv
import time
import  shutil





class  VisRender(object):
    
    def __init__(self, width=640, height=480, visible = True):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window(width= width, height=height, visible= visible)
        self.__width = width
        self.__height = height
        
        if visible:
            self.__vis.poll_events()
            self.__vis.update_renderer()
     
     
    def __del__(self):
        self.__vis.destroy_window()   
        
    @property
    def  open3d_param(self):
        return self.__open3d_param
        
    @open3d_param.setter
    def open3d_param(self, param):
        self.__open3d_param = param
        
    @property
    def camera_intrinsic(self):
        return self.__camera_intrinsic
        
    @camera_intrinsic.setter
    def camera_intrinsic(self, intrinsic):
        self.__camera_intrinsic = intrinsic
            
    
    def add_geometry(self,  data):
        #assert(data, open3d.geometry.tr)
        self.__vis.add_geometry(data)
    
    def render(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()    
    
    def get_view_point_intrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        return intrinsic

    def get_view_point_extrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic
        return extrinsic
    
      
    def convert_to_open3d_param(self, extrinsic, intrinsic= None):
        param = open3d.camera.PinholeCameraParameters()
        param.intrinsic = open3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.height =self. __height
        param.intrinsic.width =self.__width
        if  intrinsic is None:
            param.intrinsic.intrinsic_matrix = self.camera_intrinsic
        else:
            param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = extrinsic
        return param
        
        
    # def update_view_point(self,  extrinsic, intrinsic=None):
    #     ctr = self.__vis.get_view_control()
    #     param = self.convert_to_open3d_param(extrinsic)
    #     ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=False)
    #     self.__vis.update_renderer()
    
    def update_view_point(self,  extrinsic, intrinsic=None):
        param = self.convert_to_open3d_param(extrinsic)
        # print(param.intrinsic.height)
        # print(param.extrinsic)
        ctr = self.__vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        self.__vis.update_renderer()
    
    
    def capture_depth_float_buffer(self, show=False):
        depth = self.__vis.capture_depth_float_buffer(do_render=True)


        depth_convert = (np.asarray(depth)*1000).astype(np.uint16)
        if show:
            plt.imshow(depth_convert)
            plt.show()
        return depth_convert
    
    def draw_camera(self, intrinsic, extrinsic, scale=1, color=None):
        # intrinsics
        K = intrinsic

        # convert extrinsics matrix to rotation and translation matrix
        extrinsic = np.linalg.inv(extrinsic)
        R = extrinsic[0:3,0:3]
        t = extrinsic[0:3,3]

        width = self.__width
        height = self.__height

        geometries = draw_camera(K, R, t, width, height, scale, color)
        for g in geometries:
            self.add_geometry(g)
    
    def run(self):
        self.__vis.run()
            
    def destroy_window(self):
        self.__vis.destroy_window()
        
#
# Auxiliary funcions
#
def draw_camera(K, R, t, width, height, scale=1, color=None):
    """ Create axis, plane and pyramid geometries in Open3D format
    :   param K     : calibration matrix (camera intrinsics)
    :   param R     : rotation matrix
    :   param t     : translation
    :   param width : image width
    :   param height: image height
    :   param scale : camera model scale
    :   param color : color of the image plane and pyramid lines
    :   return      : camera model geometries (axis, plane and pyramid)
    """

    # default color
    if color is None:
        color = [0.8, 0.2, 0.8]

    # camera model scale
    s = 1 / scale

    # intrinsics
    Ks = np.array([[K[0, 0] * s,            0, K[0,2]],
                   [          0,  K[1, 1] * s, K[1,2]],
                   [          0,            0, K[2,2]]])
    Kinv = np.linalg.inv(Ks)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = create_coordinate_frame(T, scale=scale*0.5)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1],
    ]

    # pixel to camera coordinate system
    points = [scale * Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.transform(T)
    plane.translate(R @ [points[1][0], points[1][1], scale])

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]


def create_coordinate_frame(T, scale=0.25):
    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.transform(T)
    return frame


def render_depth(processed_dir):
    meshPath = os.path.join(processed_dir, 'fusion_mesh.ply')
    images_dir = os.path.join(processed_dir, 'images')
    #image_masks_dir = os.path.join(processed_dir , 'image_masks')
    rendered_images_dir = os.path.join(processed_dir, 'rendered_images')
    

    if not os.path.exists(rendered_images_dir):
        os.mkdir(rendered_images_dir)
    
    mesh = open3d.io.read_triangle_mesh(meshPath)
    #open3d.visualization.draw_geometries([mesh])

    width = 640
    height = 480
    fx = 1035.4066400628542
    fy = 1164.782443530703
    cx = 318.33921710833334
    cy = 237.3960612108707
    intrinsic = np.array([[fx, 0, cx],
                                              [0, fy, cy], 
                                              [0,  0,  1]], dtype = np.float64)
    
    # extrinsic = np.array([
    #      [1.43002283e-01   ,  -9.43453161e-01  ,   -2.99076043e-01  ,   -1.23520691e-02],
    #      [-9.60302442e-01  ,   -2.05394905e-01 ,    1.88764808e-01  ,   -5.31265303e-02],
    #      [-2.39519450e-01  ,   2.60209656e-01  ,   -9.35372315e-01   ,  1.02108623e+00 ],   
    #      [0.00000000e+00  ,   0.00000000e+00 ,    0.00000000e+00    , 1.00000000e+00]], dtype=np.float64)

    start_time = time.time()
    
    vis = VisRender(width=width, height=height, visible= False)
    
    # add mesh
    vis.add_geometry(mesh)
    
    # update view
    vis.camera_intrinsic = intrinsic
    
    #extrinsic = np.linalg.inv(extrinsic)
    
    # load the poses
    posefile = os.path.join(images_dir, 'pose.npy')
    poses = np.load(posefile)
    num = poses.shape[0]
    
    logging_rate = 50
    counter = 0
    
    for j in range(1,num+1):
        #pose = np.linalg.inv(poses[idx-1])
        if (counter % logging_rate) == 0:
            print ("Rendering mask for pose %d of %d" %(counter + 1, num))
        pose = poses[j-1]
        vis.update_view_point(extrinsic=pose)
        #ex = vis.get_view_point_extrinsics()
        #print(ex)
        depth = vis.capture_depth_float_buffer(show=False)
        depth_img_f = np.array(depth,  dtype=np.int32)
        idx = depth_img_f > 0
        mask = np.zeros(np.shape(depth_img_f))
        mask[idx] = 1
        visible_mask = mask*255
        

        depth_fileName = "%06i_%s.png" % (j, "depth")
    

        cv.imwrite(os.path.join( rendered_images_dir ,  depth_fileName), depth)
        
        counter+=1
        # draw camera
        #vis.draw_camera(intrinsic, pose, scale=0.5, color=[0.8, 0.2, 0.8])
        
    end_time = time.time()
    print("rendering masks took %d seconds" %(end_time - start_time) )

    del vis
    return 0

def main(processed_dir):
    meshPath = os.path.join(processed_dir, 'fusion_mesh_foreground.ply')
    images_dir = os.path.join(processed_dir, 'images')
    image_masks_dir = os.path.join(processed_dir , 'image_masks')
    rendered_images_dir = os.path.join(processed_dir, 'rendered_images')
    
    if not os.path.exists(image_masks_dir):
        os.mkdir(image_masks_dir)
    if not os.path.exists(rendered_images_dir):
        os.mkdir(rendered_images_dir)
    
    mesh = open3d.io.read_triangle_mesh(meshPath)
    #open3d.visualization.draw_geometries([mesh])

    width = 640
    height = 480
    fx = 1035.4066400628542
    fy = 1164.782443530703
    cx = 318.33921710833334
    cy = 237.3960612108707
    intrinsic = np.array([[fx, 0, cx],
                                              [0, fy, cy], 
                                              [0,  0,  1]], dtype = np.float64)
    
    # extrinsic = np.array([
    #      [1.43002283e-01   ,  -9.43453161e-01  ,   -2.99076043e-01  ,   -1.23520691e-02],
    #      [-9.60302442e-01  ,   -2.05394905e-01 ,    1.88764808e-01  ,   -5.31265303e-02],
    #      [-2.39519450e-01  ,   2.60209656e-01  ,   -9.35372315e-01   ,  1.02108623e+00 ],   
    #      [0.00000000e+00  ,   0.00000000e+00 ,    0.00000000e+00    , 1.00000000e+00]], dtype=np.float64)

    start_time = time.time()
    
    vis = VisRender(width=width, height=height, visible= False)
    
    # add mesh
    vis.add_geometry(mesh)
    
    # update view
    vis.camera_intrinsic = intrinsic
    
    #extrinsic = np.linalg.inv(extrinsic)
    
    # load the poses
    posefile = os.path.join(images_dir, 'pose.npy')
    poses = np.load(posefile)
    num = poses.shape[0]
    
    logging_rate = 50
    counter = 0
    
    for j in range(1,num+1):
        #pose = np.linalg.inv(poses[idx-1])
        if (counter % logging_rate) == 0:
            print ("Rendering mask for pose %d of %d" %(counter + 1, num))
        pose = poses[j-1]
        vis.update_view_point(extrinsic=pose)
        #ex = vis.get_view_point_extrinsics()
        #print(ex)
        depth = vis.capture_depth_float_buffer(show=False)
        depth_img_f = np.array(depth,  dtype=np.int32)
        idx = depth_img_f > 0
        mask = np.zeros(np.shape(depth_img_f))
        mask[idx] = 1
        visible_mask = mask*255
        
        mask_fileName = "%06i_%s.png" % (j, "mask")
        visible_mask_fileName = "%06i_%s.png" % (j, "visible_mask")
        depth_fileName = "%06i_%s.png" % (j, "depth_cropped")
    
        cv.imwrite( os.path.join(image_masks_dir,  mask_fileName), mask)
        cv.imwrite( os.path.join(image_masks_dir,  visible_mask_fileName), visible_mask)
        cv.imwrite(os.path.join( rendered_images_dir ,  depth_fileName), depth)
        
        counter+=1
        # draw camera
        #vis.draw_camera(intrinsic, pose, scale=0.5, color=[0.8, 0.2, 0.8])
        
    end_time = time.time()
    print("rendering masks took %d seconds" %(end_time - start_time) )

    del vis
    return 0

if __name__ == "__main__":
    processed_dir = "/home/cyn/dataset/dense-net-entire/pdc/logs_proto/000111_5/processed"
    render_depth(processed_dir)

    





    


