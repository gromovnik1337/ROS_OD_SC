#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# This script has been initiated with the goal of making an inference based on Livox Mid 40 range image
# Work initiated by: Vice, 03.09.2021

import torch
import torch.backends.cudnn as cudnn
import imp
import time
import __init__ as booger
import cv2
import os
import numpy as np

# SqueezeSegV3 specific imports
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.trainer import *
from tasks.semantic.postproc.KNN import KNN


class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir, livox_scan): # User class constructor
    # Parameters
    self.ARCH = ARCH # Arch yaml config file 
    self.DATA = DATA # Data yaml config file
    self.datadir = datadir # LiDAR data - raw
    self.logdir = logdir # Output folder
    self.modeldir = modeldir # Pre-trained model folder
    self.livox_scan = livox_scan

    # Get the data
    # This just loads the parser.py as a python module
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser_for_one.py')

    self.parser = parserModule.Parser(root=self.datadir,
                                      # This is what determines the behavior of the parser
                                      # It expects to load the data to make the inference (test), not training or validation!
                                      train_sequences=None,
                                      valid_sequences=None,
                                      test_sequences=self.DATA["split"]["sample"], 
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      livox_scan = self.livox_scan,
                                      gt=False, # Ground truth - include labels?
                                      shuffle_train=False)

    # Concatenate the encoder and the head
    # This essentially imports the architecture of the model and creates a model object so that the pre-trained parameters,
    # weights etc. could be imported and the inference could be made.
    # Inputs: arch config file of the model, get_n_classes() output from parser and the model directory
    with torch.no_grad():
      self.model = Segmentator(self.ARCH, 
                               self.parser.get_n_classes(), # Number of classes acquired from parsing the KITTI data set file 
                               self.modeldir)

    # Use KNN post processing - depends on the parameters in arch config file
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # Use GPU for inference?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    # Make a subset and conduct inference on it
    self.infer_subset(loader=self.parser.get_test_set(), # This is already a range image!
                      to_orig_fn=self.parser.to_original) # Puts the labels in original values
    print('Finished Infering')
    return

  def infer_subset(self, loader, to_orig_fn):
    # Switch to evaluate mode - from the loaded model
    self.model.eval()

    # Empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()

          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        proj_output, _, _, _, _ = self.model(proj_in, proj_mask)
        proj_argmax = proj_output[0].argmax(dim=0)

        if self.post:
          # KNN postproc
          unproj_argmax = self.post(proj_range,
                                    unproj_range,
                                    proj_argmax,
                                    p_x,
                                    p_y)
        else:
          # Put in original pointcloud using indexes
          unproj_argmax = proj_argmax[p_y, p_x]

        if torch.cuda.is_available():
         torch.cuda.synchronize()

        print("Infered seq", path_seq, "scan", path_name,
              "in", time.time() - end, "sec")
        
        end = time.time()

        # Save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)
        depth = (cv2.normalize(proj_in[0][0].cpu().numpy(), None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        print(depth.shape, proj_mask.shape,proj_argmax.shape)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * proj_mask[0].cpu().numpy()[..., None]
         # make label prediction
        pred_color = self.parser.to_color((proj_argmax.cpu().numpy() * proj_mask[0].cpu().numpy()).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        print(path)
        cv2.imwrite(path[:-6]+'.png',out_img)


