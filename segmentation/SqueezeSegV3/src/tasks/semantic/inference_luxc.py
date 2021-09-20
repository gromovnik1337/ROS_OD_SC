#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# This script has been initiated with the goal of making an inference based on Livox Mid 40 range image
# Work initiated by: Vice, 03.09.2021

import argparse
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

# Import the user script to execute the inference
from tasks.semantic.modules.user_luxc import *

import open3d as o3d

if __name__ == '__main__':

  # Livox scan
  livox_raw_pcd = o3d.io.read_point_cloud("./luxc_data/cropped_210920_subsampled.pcd", print_progress = True) # Load the point cloud
  livox_scan = np.asarray(livox_raw_pcd.points) # Convert to numpy 

  # Get the location of the sample data, output folder and the trained model.
  parser = argparse.ArgumentParser("./inference_luxc.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      default = "../../../sample_data/", 
      help='Dataset to sample'
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=  '../../../sample_output/',
      help='Directory to put the predictions. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--model', '-m',
      type=str,
      default= '../../../../SSGV3-21-20210701T140552Z-001/SSGV3-21/',
      help='Directory to get the trained model.'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # Print the summary of the inference that is to be conducted
  print("----------")
  print("Making a segmentation inference with the following parameters:")
  print("dataset", FLAGS.dataset)
  print("log", FLAGS.log)
  print("model", FLAGS.model)
  print("----------\n")


  # Open the arch config file of the pre-trained model
  try:
    print("Opening arch config file from %s" % FLAGS.model)
    ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # Open data config file of the pre-trained model
  try:
    print("Opening data config file from %s" % FLAGS.model)
    DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  # Create the output folder
  try:
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
    os.makedirs(os.path.join(FLAGS.log, "sequences"))

    for seq in DATA["split"]["sample"]:
      seq = '{0:02d}'.format(int(seq))
      print("sample_list",seq)
      os.makedirs(os.path.join(FLAGS.log,"sequences", str(seq)))
      os.makedirs(os.path.join(FLAGS.log,"sequences", str(seq), "predictions"))

  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    raise

  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  # Does the model folder exist?
  if os.path.isdir(FLAGS.model):
    print("model folder exists! Using model from %s" % (FLAGS.model))
  else:
    print("model folder doesnt exist! Can't infer...")
    quit()


  # --------------------------------  
  # Main part of the script - create the user object and infer dataset
  user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, livox_scan)
  user.infer() # Method from the User class
