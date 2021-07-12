import os 
import sys 
import random 
from datetime import datetime
import logging
import numpy 

import  importlib
import importlib.util

from modules.utils.utils import  getDictFromYamlFilename


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


_ENV_SETUP_DONE = False

# from  https://github.com/facebookresearch/detectron2/blob/03c70397ace4f03d85e99ce0241f852f65d1eb16/detectron2/utils/env.py
def setup_environment():
    
    
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
    
    
    custom_module_path = os.environ.get("DVC_ENV_MODULE")
    
    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass
    

def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """

    if custom_module.endswith(".py"):
        module = _import_file("dvc.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(custom_module)
    module.setup_environment()



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