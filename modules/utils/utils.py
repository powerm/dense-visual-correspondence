from __future__ import print_function

import sys
import os
import time
import yaml
import numpy as np
from yaml import CLoader
import datetime
import random
import  modules.utils.transformations as transformations

def getDenseCorrespondenceSourceDir():
    return os.getenv("DC_SOURCE_DIR")

def getSpartanSourceDir():
    return os.getenv("SPARTAN_SOURCE_DIR")

def get_data_dir():
    return os.getenv("DATA_DIR")

def getPdcPath():
    """
    For backwards compatibility
    """
    return get_data_dir()

def get_defaults_config():
    dc_source_dir = getDenseCorrespondenceSourceDir()
    default_config_file = os.path.join(dc_source_dir, 'config', 'defaults.yaml')

    return getDictFromYamlFilename(default_config_file)

def set_cuda_visible_devices(gpu_list):
    """
    Sets CUDA_VISIBLE_DEVICES environment variable to only show certain gpus
    If gpu_list is empty does nothing
    :param gpu_list: list of gpus to set as visible
    :return: None
    """

    if len(gpu_list) == 0:
        print("using all CUDA gpus")
        return

    cuda_visible_devices = ""
    for gpu in gpu_list:
        cuda_visible_devices += str(gpu) + ","

    print("setting CUDA_VISIBLE_DEVICES = ", cuda_visible_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def getQuaternionFromDict(d):
    quat = None
    quatNames = ['orientation', 'rotation', 'quaternion']
    for name in quatNames:
        if name in d:
            quat = d[name]


    if quat is None:
        raise ValueError("Error when trying to extract quaternion from dict, your dict doesn't contain a key in ['orientation', 'rotation', 'quaternion']")

    return quat

def compute_angle_between_quaternions(q, r):
    """
    Computes the angle between two quaternions.

    theta = arccos(2 * <q1, q2>^2 - 1)

    See https://math.stackexchange.com/questions/90081/quaternion-distance
    :param q: numpy array in form [w,x,y,z]. As long as both q,r are consistent it doesn't matter
    :type q:
    :param r:
    :type r:
    :return: angle between the quaternions, in radians
    :rtype:
    """

    theta = 2*np.arccos(2 * np.dot(q,r)**2 - 1)
    return theta

def dictFromPosQuat(pos, quat):
    d = dict()
    d['translation'] = dict()
    d['translation']['x'] = pos[0]
    d['translation']['y'] = pos[1]
    d['translation']['z'] = pos[2]

    d['quaternion'] = dict()
    d['quaternion']['w'] = quat[0]
    d['quaternion']['x'] = quat[1]
    d['quaternion']['y'] = quat[2]
    d['quaternion']['z'] = quat[3]

    return d

def get_current_YYYY_MM_DD_hh_mm_ss():
    """
    Returns a string identifying the current:
    - year, month, day, hour, minute, second

    Using this format:

    YYYY-MM-DD-hh-mm-ss

    For example:

    2018-04-07-19-02-50

    Note: this function will always return strings of the same length.

    :return: current time formatted as a string
    :rtype: string

    """

    now = datetime.datetime.now()
    string =  "%0.4d-%0.2d-%0.2d-%0.2d-%0.2d-%0.2d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    return string

def getDictFromYamlFilename(filename):
    """
    Read data from a YAML files
    """
    with open(filename) as f:
        return yaml.load(f.read(), Loader=CLoader)
        #return yaml.load(f.read())

def saveToYaml(data, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def homogenous_transform_from_dict(d):
    """
    Returns a transform from a standard encoding in dict format
    :param d:
    :return:
    """
    pos = [0]*3
    pos[0] = d['translation']['x']
    pos[1] = d['translation']['y']
    pos[2] = d['translation']['z']

    quatDict = getQuaternionFromDict(d)
    quat = [0]*4
    quat[0] = quatDict['w']
    quat[1] = quatDict['x']
    quat[2] = quatDict['y']
    quat[3] = quatDict['z']

    transform_matrix = transformations.quaternion_matrix(quat)
    transform_matrix[0:3,3] = np.array(pos)

    return transform_matrix


def convert_to_absolute_path(path):
    """
    Converts a potentially relative path to an absolute path by pre-pending the home directory
    :param path: absolute or relative path
    :type path: str
    :return: absolute path
    :rtype: str
    """

    if os.path.isdir(path):
        return path


    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, path)

def convert_data_relative_path_to_absolute_path(path, assert_path_exists=False):
    """
    Expands a path that is relative to the DC_DATA_DIR
    returned by `get_data_dir()`.

    If the path is already an absolute path then just return the path
    :param path:
    :type path:
    :param assert_path_exists: if you know this path should exist, then try to resolve it using a backwards compatibility check
    :return:
    :rtype:
    """

    if os.path.isabs(path):
        return path

    full_path = os.path.join(get_data_dir(), path)

    if assert_path_exists:
        if not os.path.exists(full_path):
            # try a backwards compatibility check for old style
            # "code/data_volume/pdc/<path>" rather than <path>
            start_path = "dataset/dense-net-entire/pdc"
            rel_path = os.path.relpath(path, start_path)
            full_path = os.path.join(get_data_dir(), rel_path)
        
        if not os.path.exists(full_path):
            raise ValueError("full_path %s not found, you asserted that path exists" %(full_path))


    return full_path

def get_unique_string():
    """
    Returns a unique string based on current date and a random number
    :return:
    :rtype:
    """

    string = get_current_YYYY_MM_DD_hh_mm_ss() + "_" + str(random.randint(0,1000))
    return string

class CameraIntrinsics(object):
    """
    Useful class for wrapping camera intrinsics and loading them from a
    camera_info.yaml file
    """
    def __init__(self, cx, cy, fx, fy, width, height):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.width = width
        self.height = height

        self.K = self.get_camera_matrix()

    def get_camera_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0,0,1]])

    @staticmethod
    def from_yaml_file(filename):
        config = getDictFromYamlFilename(filename)

        fx = config['camera_matrix']['data'][0]
        cx = config['camera_matrix']['data'][2]

        fy = config['camera_matrix']['data'][4]
        cy = config['camera_matrix']['data'][5]

        width = config['image_width']
        height = config['image_height']

        return CameraIntrinsics(cx, cy, fx, fy, width, height)



if __name__ == '__main__':

    filename = "/home/cyn/ws_moveit/src/fusion_server/config/poses.yaml"
    #filename = "/home/cyn/ws_moveit/src/camera_config/config/kinectDK/master/camera_info.yaml"

    di =getDictFromYamlFilename(filename)
    print(di)