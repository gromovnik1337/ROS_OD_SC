#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# This script has been initiated with the goal of making an inference based on Livox Mid 40 range image
# Work initiated by: Vice, 03.09.2021


import argparse
import yaml
from shutil import copyfile
import os
import shutil

import torch
import torch.backends.cudnn as cudnn
import imp
import time
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from parser_for_one import *
import open3d as o3d

# SqueezeSegV3 specific imports
#from tasks.semantic.modules.segmentator import *
#from tasks.semantic.modules.trainer import *
#from tasks.semantic.postproc.KNN import KNN

# Livox scan
#livox_raw_pcd = o3d.io.read_point_cloud("./data/lidar_screenshot_clean.pcd", print_progress = True) # Load the point cloud
livox_raw_pcd = o3d.io.read_point_cloud("./data/cropped_1.pcd", print_progress = True) # Load the point cloud
livox_scan = np.asarray(livox_raw_pcd.points) # Convert to numpy 

print("At the entry, livox scan is in shape: ", np.shape(livox_scan))

# Parameters
datadir = "../sample_data/" # LiDAR data - raw
logdir = '../sample_output/' # Output folder
modeldir = '../../SSGV3-21-20210701T140552Z-001/SSGV3-21/' # Pre-trained model folder

# Does the model folder exist?
if os.path.isdir(modeldir):
    print("model folder exists! Using model from %s" % (modeldir))
else:
    print("model folder doesnt exist! Can't infer...")
    quit()

# Open the arch config file of the pre-trained model
try:
    print("Opening arch config file from %s" % modeldir)
    ARCH = yaml.safe_load(open(modeldir + "/arch_cfg.yaml", 'r'))
except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

# Open data config file of the pre-trained model
try:
    print("Opening data config file from %s" % modeldir)
    DATA = yaml.safe_load(open(modeldir + "/data_cfg.yaml", 'r'))
except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

# Create the output folder
try:
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    os.makedirs(os.path.join(logdir, "sequences"))

    for seq in DATA["split"]["sample"]:
        seq = '{0:02d}'.format(int(seq))
        print("sample_list",seq)
        os.makedirs(os.path.join(logdir,"sequences", str(seq)))
        os.makedirs(os.path.join(logdir,"sequences", str(seq), "predictions"))

except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    raise

except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

# Create a parser and get the data
parser = Parser(root=datadir,
                                  # This is what determines the behavior of the parser
                                  # It expects to load the data to make the inference (test), not training or validation!
                                  train_sequences=None,
                                  valid_sequences=None,
                                  test_sequences=DATA["split"]["sample"], 
                                  labels=DATA["labels"],
                                  color_map=DATA["color_map"],
                                  learning_map=DATA["learning_map"],
                                  learning_map_inv=DATA["learning_map_inv"],
                                  sensor=ARCH["dataset"]["sensor"],
                                  max_points=ARCH["dataset"]["max_points"],
                                  batch_size=1,
                                  workers=ARCH["train"]["workers"],
                                  livox_scan = livox_scan,
                                  gt=False, # Ground truth - include labels?
                                  shuffle_train=False)

test = parser.get_test_set() # This returns one DataLoader Object - it is a Python iterator returning tuples
test_size = parser.get_test_size() # Number of samples in the DataLoader object. Every sample (tuple) contains all the data from one LiDAR scan
test_iterator = iter(test) # This returns an iterator for the DataLoader Object.

first = next(test_iterator) # This returns a first element of the iterator. It contains 15 elements, listed below:
"""
proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name,
proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points
"""

# Projection 
proj = first[0].numpy() # Convert the projection and projection mask to numpy
proj_mask = first[1].numpy()

proj = np.squeeze(proj) # Remove ones
proj_mask = np.squeeze(proj_mask)

print(proj_mask)

# Plot the projections

plt.figure()
plt.title("Projection mask")
plt.imshow(proj_mask)

plt.figure()
plt.title("Projection layer 1")
plt.imshow(proj[0])

plt.figure()
plt.title("Projection layer 2")
plt.imshow(proj[1])

plt.figure()
plt.title("Projection layer 3")
plt.imshow(proj[2])

plt.figure()
plt.title("Projection layer 4")
plt.imshow(proj[3])

plt.figure()
plt.title("Projection layer 5")
plt.imshow(proj[4])

plt.show()


