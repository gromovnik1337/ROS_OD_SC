import os
import numpy as np
import torch
from torch.utils.data import Dataset

from common.laserscan_for_one import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

class SemanticKitti(Dataset):
  """This class creates a PyTorch Dataset object that can be loaded into the model.

  Args:
      Dataset ([class]): Dataset class to inherit from.
  """

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               livox_scan,          # Livox scan data - added by Vice 10.09.2021
               max_points=150000,   # max number of points present in dataset
               gt=False):            # send ground truth?

    # Parameters of the LiDAR scans
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.livox_scan = livox_scan
    self.max_points = max_points
    self.gt = gt

    # Get the number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)
    print(self.nclasses)

    # Sanity checks
    # Make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # Make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # Make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # Make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # Make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # Placeholder for filenames
    self.scan_files = []
    self.label_files = []

    # Fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # To string
      seq = '{0:02d}'.format(int(seq))

      print("parsing seq {}".format(seq))

      # Get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")
      label_path = os.path.join(self.root, seq, "labels")

      # Get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]

      # Check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # Extend list
      self.scan_files.extend(scan_files)
      self.label_files.extend(label_files)

    # Sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()

    print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                    self.sequences))

  def __getitem__(self, index):
    # Get item in tensor shape
    scan_file = self.scan_files[index]
    if self.gt:
      label_file = self.label_files[index]

    # Create a LiDAR laser scan object, with (semantic) or without labels
    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                      # Following parameters are from the KITTI data set format
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

  
    # Open and obtain scan - general case
    # Make the same thing for the Livox scan, keep the label data from the dummy Velodyne scans
    # Works onl if self.gt = False
    # ----------------------------------------------------------------------------------------------------------
    livox_range = self.livox_scan
    scan.open_scan(scan_file, True, livox_range) #TODO This Bool must go out! It determines are we using livox scan or not

    # For the semantic case
    if self.gt:
      scan.open_label(label_file)
      # Map unused classes to used classes (also for projection)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    # Make a tensor of the uncompressed data (with the max num points)
    unproj_n_points = scan.points.shape[0]

    # TODO Remove
    unproj_xyz = torch.full((unproj_n_points, 3), -1.0, dtype=torch.float) # removed self.max_points

    print("Scan points: ", np.shape(scan.points)[0])
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)

    # TODO Remove
    unproj_range = torch.full([unproj_n_points], -1.0, dtype=torch.float) # removed self.max_points

    print("unproj_range: ", scan.unproj_range.shape[0])
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)

    #  TODO Remove
    unproj_remissions = torch.full([unproj_n_points], -1.0, dtype=torch.float) # removed self.max_points

    # Changed to accomodate for the fact that remissions are taken from Velodyne data and they do not exist in Livox
    #unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.points[:unproj_n_points])
    
    # Old - works
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions[:unproj_n_points]) 

    if self.gt:
      unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
    else:
      unproj_labels = []

    # Get points and labels
    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []
    
    # TODO Remove
    proj_x = torch.full([unproj_n_points], -1, dtype=torch.long) # removed self.max_points
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
    
    # TODO Remove
    proj_y = torch.full([unproj_n_points], -1, dtype=torch.long) # removed self.max_points
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
    
    # This is where projection is defined
    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2,0,1),
                      proj_remission.unsqueeze(0).clone()])
    proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
    proj = proj * proj_mask.float()

    # Get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")

    # Returng the range image and all the relevant data
    return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, \
    proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

  def __len__(self):
    return len(self.scan_files)

  @staticmethod
  def map(label, mapdict):
    # Put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # Make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # Do the mapping
    return lut[label]


class Parser():
  # Standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               livox_scan,          # Livox scan data as numpy array, added by Vice 10.09.2021 
               gt=False,           # get gt?
               shuffle_train=True  # shuffle training set?
               ):  
    super(Parser, self).__init__()

    # If I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.livox_scan = livox_scan
    self.gt = gt
    self.shuffle_train = shuffle_train

    # Number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)
    print(self.nclasses)
    
    # Data loading code
    # Training
    if self.train_sequences:
        self.train_dataset = SemanticKitti(root=self.root,
                                       sequences=self.train_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       livox_scan = self.livox_scan,
                                       max_points=max_points,
                                       gt=self.gt)

        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
        assert len(self.trainloader) > 0
        self.trainiter = iter(self.trainloader)

    # Validation
    if self.valid_sequences:
        self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       livox_scan = self.livox_scan,
                                       max_points=max_points,
                                       gt=self.gt)

        self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
        assert len(self.validloader) > 0
        self.validiter = iter(self.validloader)

    # Inference
    if self.test_sequences: 
      self.test_dataset = SemanticKitti(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        livox_scan = self.livox_scan,
                                        sensor=self.sensor,
                                        max_points=max_points,
                                        gt=False)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)

  # Methods
  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # Put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # Put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # Put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # Put label in color
    return SemanticKitti.map(label, self.color_map)
