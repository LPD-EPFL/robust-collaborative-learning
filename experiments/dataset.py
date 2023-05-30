# coding: utf-8
###
 # @file   dataset.py
 # @author John stephan <john.Stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Dataset wrappers/helpers.
###

__all__ = ["get_default_transform", "Dataset", "make_sampler", "make_datasets",
           "batch_dataset"]

import tools

import pathlib
import random
import tempfile
import torch
import torchvision
import types
import torchvision.transforms as T
import numpy as np
import sys

#JS: import Flamby datasets
try:
  path_FLamby = '/root/FLamby'
  sys.path.append(path_FLamby)
  from flamby.datasets.fed_heart_disease import FedHeartDisease
  from flamby.datasets.fed_tcga_brca import FedTcgaBrca
except:
  pass

# ---------------------------------------------------------------------------- #
# Default image transformations

# Collection of default transforms, <dataset name> -> (<train transforms>, <test transforms>)
transforms_horizontalflip = [
  torchvision.transforms.RandomHorizontalFlip(),
  torchvision.transforms.ToTensor()]
transforms_mnist = [
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.1307,), (0.3081,))] # Transforms from "A Little is Enough" (https://github.com/moranant/attacking_distributed_learning)
transforms_cifar = [
  torchvision.transforms.RandomHorizontalFlip(),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] # Transforms from https://github.com/kuangliu/pytorch-cifar

# Per-dataset image transformations (automatically completed, see 'Dataset._get_datasets')
transforms = {
  "mnist":        (transforms_mnist, transforms_mnist),
  "fashionmnist": (transforms_horizontalflip, transforms_horizontalflip),
  "cifar10":      (transforms_cifar, transforms_cifar),
  "cifar100":     (transforms_cifar, transforms_cifar),
  "imagenet":     (transforms_horizontalflip, transforms_horizontalflip) }

#JS: Dataset names in pytorch
dict_names = {
  "mnist":        "MNIST",
  "fashionmnist": "FashionMNIST",
  "emnist":       "EMNIST",
  "cifar10":      "CIFAR10",
  "cifar100":     "CIFAR100",
  "imagenet":     "ImageNet"}

def get_default_transform(dataset, train):
  """ Get the default transform associated with the given dataset name.
  Args:
    dataset Case-sensitive dataset name, or None to get no transformation
    train   Whether the transformation is for the training set (always ignored if None is given for 'dataset')
  Returns:
    Associated default transformations (always exist)
  """
  global transforms
  # Fetch transformation
  transform = transforms.get(dataset)
  # Not found (not a torchvision dataset)
  if transform is None:
    return None
  # Return associated transform
  return torchvision.transforms.Compose(transform[0 if train else 1])

# ---------------------------------------------------------------------------- #
# Dataset loader-batch producer wrapper class

class Dataset:
  """ Dataset wrapper class.
  """

  # Default dataset root directory path
  __default_root = None

  @classmethod
  def get_default_root(self):
    """ Lazy-initialize and return the default dataset root directory path.
    Returns:
      '__default_root'
    """
    # Fast-path already loaded
    if self.__default_root is not None:
      return self.__default_root
    # Generate the default path
    self.__default_root = pathlib.Path(__file__).parent / "datasets" / "cache"
    # Warn if the path does not exist and fallback to '/tmp'
    if not self.__default_root.exists():
      tmpdir = tempfile.gettempdir()
      tools.warning(f"Default dataset root {str(self.__default_root)!r} does not exist, falling back to local temporary directory {tmpdir!r}", context="experiments")
      self.__default_root = pathlib.Path(tmpdir)
    # Return the path
    return self.__default_root

  # Map 'lower-case names' -> 'dataset class' available in PyTorch
  __datasets = None

  @classmethod
  def _get_datasets(self):
    """ Lazy-initialize and return the map '__datasets'.
    Returns:
      '__datasets'
    """
    global transforms
    # Fast-path already loaded
    if self.__datasets is not None:
      return self.__datasets
    # Initialize the dictionary
    self.__datasets = dict()
    # Populate this dictionary with TorchVision's datasets
    for name in dir(torchvision.datasets):
      if len(name) == 0 or name[0] == "_": # Ignore "protected" members
        continue
      constructor = getattr(torchvision.datasets, name)
      if isinstance(constructor, type): # Heuristic
        def make_builder(constructor, name):
          def builder(root, batch_size=None, shuffle=False, num_workers=1, dataset_post=None, *args, **kwargs):
            # Try to build the dataset instance
            data = constructor(root, *args, **kwargs)
            assert isinstance(data, torch.utils.data.Dataset), f"Internal heuristic failed: {name!r} was not a dataset name"
            # Post-process the dataset instance, if requested
            if dataset_post is not None:
              data = dataset_post(data)
            # Ensure there is at least a tensor transformation for each torchvision dataset
            if name not in transforms:
              transforms[name] = torchvision.transforms.ToTensor()
            # Wrap into a loader
            batch_size = batch_size or len(data)
            loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            # Wrap into an infinite batch sampler generator
            return make_sampler(loader)
          return builder
        self.__datasets[name.lower()] = make_builder(constructor, name)
    # Dynamically add the custom datasets from subdirectory 'datasets/'
    def add_custom_datasets(name, module, _):
      nonlocal self
      # Check if has exports, fallback otherwise
      exports = getattr(module, "__all__", None)
      if exports is None:
        tools.warning(f"Dataset module {name!r} does not provide '__all__'; falling back to '__dict__' for name discovery")
        exports = (name for name in dir(module) if len(name) > 0 and name[0] != "_")
      # Register the association 'name -> constructor' for all the datasets
      exported = False
      for dataset in exports:
        # Check dataset name type
        if not isinstance(dataset, str):
          tools.warning(f"Dataset module {name!r} exports non-string name {dataset!r}; ignored")
          continue
        # Recover instance from name
        constructor = getattr(module, dataset, None)
        # Check instance is callable (it's only an heuristic...)
        if not callable(constructor):
          continue
        # Register callable with composite name
        exported = True
        fullname = f"{name}-{dataset}"
        if fullname in self.__datasets:
          tools.warning(f"Unable to make available dataset {dataset!r} from module {name!r}, as the name {fullname!r} already exists")
          continue
        self.__datasets[fullname] = constructor
      if not exported:
        tools.warning(f"Dataset module {name!r} does not export any valid constructor name through '__all__'")
    with tools.Context("datasets", None):
      tools.import_directory(pathlib.Path(__file__).parent / "datasets", {"__package__": f"{__package__}.datasets"}, post=add_custom_datasets)
    # Return the dictionary
    return self.__datasets

  def __init__(self, data, gradient_descent=False, heterogeneity=False, numb_labels=None, distinct_datasets=False, flamby_datasets=None, gamma_similarity=None, alpha_dirichlet=None,
    nb_datapoints=None, honest_workers=None, batch_size_train=None, batch_size_test=None, name=None, root=None, *args, **kwargs):
    """ Dataset builder constructor.
    Args:
      data                  Dataset string name, (infinite) generator instance (that will be used to generate samples), or any other instance (that will then be fed as the only sample)
      gradient_descent      Boolean that is true in the case of the full gradient descent algorithm
      heterogeneity         Boolean that is true in heterogeneous setting
      numb_labels           Number of labels of the dataset in question
      distinct_datasets     Boolean that is true in setting where honest workers must have distinct datasets (e.g., privacy setting)
      flamby_datasets       String that is not None when using Flamby datasets, set to "heart" or "tcg"
      gamma_similarity      Float for distributing the datasets among honest workers
      alpha_dirichlet       Value of parameter alpha for dirichlet distribution
      nb_datapoints         Number of datapoints per honest worker in case of distinct datasets
      honest_workers        Number of honest workers in the system
      batch_size_train      Batch size used for the heterogeneous data loader
      batch_size_test       Batch size used for the testing data loader
      name                  Optional user-defined dataset name, to attach to some error messages for debugging purpose
      root                  Dataset cache root directory to use, None for default (only relevant if 'data' is a dataset name)
      ...                   Forwarded (keyword-)arguments to the dataset constructor, ignored if 'data' is not a string
    Raises:
      'TypeError' if the some of the given (keyword-)arguments cannot be used to call the dataset or loader constructor or the batch loader
    """
    # Handle different dataset types
    if isinstance(data, str): # Load sampler from available datasets

      if name is None:
        name = data
      datasets = type(self)._get_datasets()
      build = datasets.get(name, None)
      if build is None:
        raise tools.UnavailableException(datasets, name, what="dataset name")
      root = root or type(self).get_default_root()

      #JS: Load the initial dataset and targets
      try:
          loaded_transform = T.Compose(transforms[data][0])
          dataset = getattr(torchvision.datasets, dict_names[data])(root=root, train=True, download=True, transform=loaded_transform)
          targets = dataset.targets
          if isinstance(targets, list):
              targets = torch.FloatTensor(targets)
      except:
          loaded_transform = None

      #JS: Flamby datasets
      if flamby_datasets is not None and kwargs['train']:
        self.iter_dict = {}
        self.dataset_dict = {}

        for worker_id in range(honest_workers):
          if flamby_datasets == "heart":
              center_id = FedHeartDisease(center=worker_id, train=True)
          else:
              center_id = FedTcgaBrca(center=worker_id, train=True)
          dataset_worker = torch.utils.data.DataLoader(center_id, batch_size=batch_size_train, shuffle=True, num_workers=0)
          #JS: have one dataset iterator per honest worker
          self.dataset_dict[worker_id] = dataset_worker

      #JS: extreme heterogeneity setting while training
      elif heterogeneity and kwargs['train']:
        labels = range(numb_labels)
        ordered_indices = []
        for label in labels:
          label_indices = (targets == label).nonzero().tolist()
          label_indices = [item for sublist in label_indices for item in sublist]
          ordered_indices += label_indices

        self.iter_dict = {}
        self.dataset_dict = {}

        split_indices = np.array_split(ordered_indices, honest_workers)
        for worker_id in range(honest_workers):
          dataset_modified = torch.utils.data.Subset(dataset, split_indices[worker_id].tolist())
          if gradient_descent:
            #JS: Adjust batch size in case of gradient descent
            batch_size_train = len(split_indices[worker_id])
          dataset_worker = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size_train, shuffle=True)
          #JS: have one dataset iterator per honest worker
          self.dataset_dict[worker_id] = dataset_worker

      #JS: distinct datasets for honest workers with gamma similarity
      elif distinct_datasets and kwargs['train'] and gamma_similarity is not None:
        numb_samples = len(dataset.targets)
        numb_samples_iid = int(gamma_similarity * numb_samples)

        #JS: Sample gamma % of the dataset, and build homogeneous dataset
        homogeneous_dataset, _ = torch.utils.data.random_split(dataset, [numb_samples_iid, numb_samples - numb_samples_iid])

        #JS: Split the indices of the homogeneous dataset onto the honest workers
        split_indices_homogeneous = np.array_split(homogeneous_dataset.indices, honest_workers)

        #JS: Rearrange the entire dataset by sorted labels
        labels = range(numb_labels)
        ordered_indices = []
        for label in labels:
          label_indices = (targets == label).nonzero().tolist()
          label_indices = [item for sublist in label_indices for item in sublist]
          ordered_indices += label_indices
        #JS: split the (sorted) heterogeneous indices equally among the honest workers
        indices_heterogeneous = [index for index in ordered_indices if index not in homogeneous_dataset.indices]
        split_indices_heterogeneous = np.array_split(indices_heterogeneous, honest_workers)

        self.iter_dict = {}
        self.dataset_dict = {}

        for worker_id in range(honest_workers):
          homogeneous_dataset_worker = torch.utils.data.Subset(dataset, split_indices_homogeneous[worker_id])
          heterogeneous_dataset_worker = torch.utils.data.Subset(dataset, split_indices_heterogeneous[worker_id])
          concat_datasets = torch.utils.data.ConcatDataset([homogeneous_dataset_worker, heterogeneous_dataset_worker])
          if gradient_descent:
            #JS: Adjust batch size in case of gradient descent
            batch_size_train = len(split_indices_homogeneous[worker_id]) + len(split_indices_heterogeneous[worker_id])
          dataset_worker = torch.utils.data.DataLoader(concat_datasets, batch_size=batch_size_train, shuffle=True)
          #JS: have one dataset iterator per honest worker
          self.dataset_dict[worker_id] = dataset_worker

      #JS: distinct datasets for honest workers (used in privacy setting for example), homogeneous setting
      elif distinct_datasets and kwargs['train']:
        numb_samples = len(dataset.targets)
        sample_indices = list(range(numb_samples))
        random.shuffle(sample_indices)

        self.iter_dict = {}
        self.dataset_dict = {}
        if nb_datapoints is None:
          #JS: split the whole dataset equally among the honest workers
          split_indices = np.array_split(sample_indices, honest_workers)
        else:
          #JS: give every honest worker nb_datapoints samples
          split_indices = [sample_indices[i:i + nb_datapoints] for i in range(0, nb_datapoints*honest_workers, nb_datapoints)]

        for worker_id in range(honest_workers):
          dataset_modified = torch.utils.data.Subset(dataset, split_indices[worker_id])
          if gradient_descent:
            #JS: Adjust batch size in case of gradient descent
            batch_size_train = len(split_indices[worker_id])

          dataset_worker = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size_train, shuffle=True)
          #JS: have one dataset iterator per honest worker
          self.dataset_dict[worker_id] = dataset_worker

      #JS: distribute data among honest workers using Dirichlet distribution
      elif alpha_dirichlet is not None and kwargs['train']:

        def _distribute(alpha, nb_workers, nb_classes, samples_per_worker):
          """ returns number of samples of each class for each worker """
          samples_distribution = samples_per_worker * np.random.dirichlet(np.repeat(alpha, nb_classes), size=nb_workers)
          samples_distribution = samples_distribution.astype(int)
          # handle numerical crashes
          for worker in samples_distribution:
            if worker[0] < 0:
              idx_filled = np.random.randint(nb_classes)
              for idx in range(nb_classes):
                worker[idx] = int(idx == idx_filled) * samples_per_worker
          return samples_distribution

        def _draw_indices(samples_distribution, buckets_indices):
          """ returns all indices selected for each worker """
          all_indices = list()
          for distr in samples_distribution:
            worker_indices = list()
            for nb_samples, bucket in zip(distr, buckets_indices):
              samples_to_extract = min(nb_samples, len(bucket))
              worker_indices += random.sample(bucket, samples_to_extract)
            all_indices.append(worker_indices)
          return all_indices

        data_per_worker = len(targets) // honest_workers

        buckets_indices = []
        labels = range(numb_labels)
        for label in labels:
          label_indices = (targets == label).nonzero().tolist()
          label_indices = [item for sublist in label_indices for item in sublist]
          buckets_indices.append(label_indices)
        samples_distribution = _distribute(alpha=alpha_dirichlet, nb_workers=honest_workers, nb_classes=numb_labels, samples_per_worker=data_per_worker)
        all_indices = _draw_indices(samples_distribution, buckets_indices)

        self.iter_dict = {}
        self.dataset_dict = {}
        for worker_id in range(honest_workers):
          dataset_modified = torch.utils.data.Subset(dataset, all_indices[worker_id])
          if gradient_descent:
            #JS: Adjust batch size in case of gradient descent
            batch_size_train = len(all_indices[worker_id])

          dataset_worker = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size_train, shuffle=True)
          #JS: have one dataset iterator per honest worker
          self.dataset_dict[worker_id] = dataset_worker

      #JS: testing set for Flamby datasets
      elif flamby_datasets is not None and not kwargs['train']:
        concat_datasets = list()
        for worker_id in range(honest_workers):
          if flamby_datasets == "heart":
              concat_datasets.append(FedHeartDisease(center=worker_id, train=False))
          else:
              concat_datasets.append(FedTcgaBrca(center=worker_id, train=False))
        dataset_test = torch.utils.data.ConcatDataset(concat_datasets)
        self.dataset_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0)

      else:
        #JS: Either homogeneous setting, or testing set (but not Flamby)
        self._iter = build(root=root, *args, **kwargs)

    elif isinstance(data, types.GeneratorType): # Forward sampling to custom generator
      if name is None:
        name = "<generator>"
      self._iter = data
    else: # Single-batch dataset of any value
      if name is None:
        name = "<single-batch>"
      def single_batch():
        while True:
          yield data
      self._iter = single_batch()
    # Finalization
    self.name = name

  def __str__(self):
    """ Compute the "informal", nicely printable string representation of this dataset.
    Returns:
      Nicely printable string
    """
    return f"dataset {self.name}"

  def sample(self, flamby_datasets=None, config=None):
    """ Sample the next batch from this dataset.
    Args:
      flamby_datasets  String that is not None when using Flamby datasets, set to "heart" or "tcg"
      config Target configuration for the sampled tensors
    Returns:
      Next batch
    """

    if flamby_datasets is not None:
        try:
            tns = next(self.iter)
        except:
            self.iter = iter(self.dataset_test)
            tns = next(self.iter)
        if config is not None:
          tns = type(tns)(tn.to(device=config["device"], non_blocking=config["non_blocking"]) for tn in tns)
        return tns

    tns = next(self._iter)
    if config is not None:
      tns = type(tns)(tn.to(device=config["device"], non_blocking=config["non_blocking"]) for tn in tns)
    return tns

  def sample_worker(self, worker_id, config=None):
    """ Sample the next batch from the dataset of worker worker_id.
    Args:
      worker_id ID of worker
      config Target configuration for the sampled tensors
    Returns:
      Next batch
    """
    try:
        tns = next(self.iter_dict[worker_id])
    except:
        self.iter_dict[worker_id] = iter(self.dataset_dict[worker_id])
        tns = next(self.iter_dict[worker_id])
    if config is not None:
      tns = type(tns)(tn.to(device=config["device"], non_blocking=config["non_blocking"]) for tn in tns)
    return tns

  def epoch(self, config=None):
    """ Return a full epoch iterable from this dataset.
    Args:
      config Target configuration for the sampled tensors
    Returns:
      Full epoch iterable
    Notes:
      Only work for dataset based on PyTorch's DataLoader
    """
    # Assert dataset based on DataLoader
    assert isinstance(self._loader, torch.utils.data.DataLoader), "Full epoch iteration only possible for PyTorch's DataLoader-based datasets"
    # Return a full epoch iterator
    epoch = self._loader.__iter__()
    def generator():
      nonlocal epoch
      try:
        while True:
          tns = next(epoch)
          if config is not None:
            tns = type(tns)(tn.to(device=config["device"], non_blocking=config["non_blocking"]) for tn in tns)
          yield tns
      except StopIteration:
        return
    return generator()

# ---------------------------------------------------------------------------- #
# Dataset helpers

def make_sampler(loader):
  """ Infinite sampler generator from a dataset loader.
  Args:
    loader Dataset loader to use
  Yields:
    Sample, forever (transparently iterating the given loader again and again)
  """
  itr = None
  while True:
    for _ in range(2):
      # Try sampling the next batch
      if itr is not None:
        try:
          yield next(itr)
          break
        except StopIteration:
          pass
      # Ask loader for a new iteration
      itr = iter(loader)
    else:
      raise RuntimeError(f"Unable to sample a new batch from dataset {name!r}")

def make_datasets(dataset, gradient_descent=False, heterogeneity=False, numb_labels=None, flamby_datasets=None, distinct_datasets=False, gamma_similarity=None, alpha_dirichlet=None,
    nb_datapoints=None, honest_workers=None, train_batch=None, test_batch=None, train_transforms=None, test_transforms=None, num_workers=1,
    train_post=None, test_post=None, **custom_args):
  """ Helper to make new instances of training and testing datasets.
  Args:
    dataset             Case-sensitive dataset name
    gradient_descent    Boolean that is true in the case of the full gradient descent algorithm
    heterogeneity       Boolean that is true in heterogeneous setting
    numb_labels         Number of labels of dataset
    flamby_datasets     String that is not None when using Flamby datasets, set to "heart" or "tcg"
    distinct_datasets   Boolean that is true in setting where honest workers must have distinct datasets (e.g., privacy setting)
    gamma_similarity    Float for distributing the datasets among honest workers
    alpha_dirichlet     Value of parameter alpha for dirichlet distribution
    nb_datapoints       Number of datapoints per honest worker in case of distinct datasets
    honest_workers      Number of honest workers in the system
    train_batch         Training batch size, None or 0 for maximum possible
    test_batch          Testing batch size, None or 0 for maximum possible
    train_transforms    Transformations to apply on the training set, None for default for the given dataset
    test_transforms     Transformations to apply on the testing set, None for default for the given dataset
    num_workers         Positive number of workers for each of the training and testing datasets, or tuple for each of them
    train_post          Training dataset instance post-processing closure
    test_post           Testing dataset instance post-processing closure
    ...                 Additional dataset-dependent keyword-arguments
  Returns:
    Training dataset, testing dataset
  """
  # Pre-process arguments
  train_transforms = train_transforms or get_default_transform(dataset, True)
  test_transforms = test_transforms or get_default_transform(dataset, False)
  num_workers_errmsg = "Expected either a positive int or a tuple of 2 positive ints for parameter 'num_workers'"
  if isinstance(num_workers, int):
    assert num_workers > 0, num_workers_errmsg
    train_workers = test_workers = num_workers
  else:
    assert isinstance(num_workers, tuple) and len(num_workers) == 2, num_workers_errmsg
    train_workers, test_workers = num_workers
    assert isinstance(train_workers, int) and train_workers > 0, num_workers_errmsg
    assert isinstance(test_workers, int)  and test_workers > 0,  num_workers_errmsg
  # Make the datasets
  trainset = Dataset(dataset, gradient_descent=gradient_descent, heterogeneity=heterogeneity, numb_labels=numb_labels,
        flamby_datasets=flamby_datasets, distinct_datasets=distinct_datasets, gamma_similarity=gamma_similarity,
        alpha_dirichlet=alpha_dirichlet, nb_datapoints=nb_datapoints, honest_workers=honest_workers, train=True, download=True,
        batch_size=train_batch, batch_size_train=train_batch, shuffle=True, num_workers=train_workers, transform=train_transforms,
        dataset_post=train_post, **custom_args)
  testset = Dataset(dataset, train=False, download=False, batch_size=test_batch, batch_size_test=test_batch, flamby_datasets=flamby_datasets,
      shuffle=False, num_workers=test_workers, honest_workers=honest_workers,transform=test_transforms,
      dataset_post=test_post, **custom_args)
  # Return the datasets
  return trainset, testset

def batch_dataset(inputs, labels, train=False, batch_size=None, split=0.75):
  """ Batch a given raw (tensor) dataset into either a training or testing infinite sampler generators.
  Args:
    inputs     Tensor of positive dimension containing input data
    labels     Tensor of same shape as 'inputs' containing expected output data
    train      Whether this is for training (basically adds shuffling)
    batch_size Training batch size, None (or 0) for maximum batch size
    split      Fraction of datapoints to use in the train set if < 1, or #samples in the train set if ≥ 1
  Returns:
    Training or testing set infinite sampler generator (with uniformly sampled batches),
    Test set infinite sampler generator (without random sampling)
  """
  def train_gen(inputs, labels, batch):
    cursor = 0
    datalen = len(inputs)
    shuffle = list(range(datalen))
    random.shuffle(shuffle)
    while True:
      end = cursor + batch
      if end > datalen:
        select = shuffle[cursor:]
        random.shuffle(shuffle)
        select += shuffle[:(end % datalen)]
      else:
        select = shuffle[cursor:end]
      yield inputs[select], labels[select]
      cursor = end % datalen
  def test_gen(inputs, labels, batch):
    cursor = 0
    datalen = len(inputs)
    while True:
      end = cursor + batch
      if end > datalen:
        select = list(range(cursor, datalen)) + list(range(end % datalen))
        yield inputs[select], labels[select]
      else:
        yield inputs[cursor:end], labels[cursor:end]
      cursor = end % datalen
  # Split dataset
  dataset_len = len(inputs)
  if dataset_len < 1 or len(labels) != dataset_len:
    raise RuntimeError(f"Invalid or different input/output tensor lengths, got {len(inputs)} for inputs, got {len(labels)} for labels")
  split_pos = min(max(1, int(dataset_len * split)) if split < 1 else split, dataset_len - 1)
  # Make and return generator according to flavor
  if train:
    train_len = split_pos
    batch_size = min(batch_size or train_len, train_len)
    return train_gen(inputs[:split_pos], labels[:split_pos], batch_size)
  else:
    test_len = dataset_len - split_pos
    batch_size = min(batch_size or test_len, test_len)
    return test_gen(inputs[split_pos:], labels[split_pos:], batch_size)
