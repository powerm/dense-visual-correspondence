import numpy as np
import os
import fnmatch
import pandas as pd
import sklearn.metrics as sm
import scipy.stats as ss
import matplotlib.pyplot as plt

import modules.utils.utils as utils
# utils.add_dense_correspondence_to_python_path()

from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluationPlotter as DCEP


folder_name = "fox"
path_to_nets = os.path.join("dataset/dense-net-entire/pdc/trained_models", folder_name)
path_to_nets = utils.convert_to_absolute_path(path_to_nets)
all_nets = sorted(os.listdir(path_to_nets))
nets_to_plot = []

nets_list = ["fox_M_background_0.500_3",  "fox_M_background_1.000_3"]
for net in nets_list:
    nets_to_plot.append(os.path.join(folder_name,net))


# Training 
#Evaluate the network on the training scenes. Correspondences are all within scene
p = DCEP()
#dc_source_dir = utils.getDenseCorrespondenceSourceDir()
data_source_dir = utils.get_data_dir()

network_name = nets_to_plot[0]
path_to_csv = os.path.join(data_source_dir, "trained_models", network_name, "analysis/train/data.csv")
fig_axes = DCEP.run_on_single_dataframe(path_to_csv, label=network_name, save=False)

for network_name in nets_to_plot[1:]:
    path_to_csv = os.path.join(data_source_dir, "trained_models", network_name, "analysis/train/data.csv")
    fig_axes = DCEP.run_on_single_dataframe(path_to_csv, label=network_name, previous_fig_axes=fig_axes, save=False)

_, axes = fig_axes
# axes[0].set_title("Training Set")
plt.show()