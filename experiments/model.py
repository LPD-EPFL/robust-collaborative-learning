# coding: utf-8
###
 # @file   model.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Model wrappers/helpers.
###

__all__ = ["Model"]

import tools

import pathlib
import torch
import torchvision
import types
import copy

from .configuration import Configuration

# ---------------------------------------------------------------------------- #
# Model wrapper class

class Model:
  """ Model wrapper class.
  """

  # Map 'lower-case names' -> 'model constructor' available in PyTorch
  __models = None

  # Map 'lower-case names' -> 'tensor initializer' available in PyTorch
  __inits = None

  @classmethod
  def _get_models(self):
    """ Lazy-initialize and return the map '__models'.
    Returns:
      '__models'
    """
    # Fast-path already loaded
    if self.__models is not None:
      return self.__models
    # Initialize the dictionary
    self.__models = dict()
    # Populate this dictionary with TorchVision's models
    for name in dir(torchvision.models):
      if len(name) == 0 or name[0] == "_": # Ignore "protected" members
        continue
      builder = getattr(torchvision.models, name)
      if isinstance(builder, types.FunctionType): # Heuristic
        self.__models[f"torchvision-{name.lower()}"] = builder
    # Dynamically add the custom models from subdirectory 'models/'
    def add_custom_models(name, module, _):
      nonlocal self
      # Check if has exports, fallback otherwise
      exports = getattr(module, "__all__", None)
      if exports is None:
        tools.warning(f"Model module {name!r} does not provide '__all__'; falling back to '__dict__' for name discovery")
        exports = (name for name in dir(module) if len(name) > 0 and name[0] != "_")
      # Register the association 'name -> constructor' for all the models
      exported = False
      for model in exports:
        # Check model name type
        if not isinstance(model, str):
          tools.warning(f"Model module {name!r} exports non-string name {model!r}; ignored")
          continue
        # Recover instance from name
        constructor = getattr(module, model, None)
        # Check instance is callable (it's only an heuristic...)
        if not callable(constructor):
          continue
        # Register callable with composite name
        exported = True
        fullname = f"{name}-{model}"
        if fullname in self.__models:
          tools.warning(f"Unable to make available model {model!r} from module {name!r}, as the name {fullname!r} already exists")
          continue
        self.__models[fullname] = constructor
      if not exported:
        tools.warning(f"Model module {name!r} does not export any valid constructor name through '__all__'")
    with tools.Context("models", None):
      tools.import_directory(pathlib.Path(__file__).parent / "models", {"__package__": f"{__package__}.models"}, post=add_custom_models)
    # Return the dictionary
    return self.__models

  @classmethod
  def _get_inits(self):
    """ Lazy-initialize and return the map '__inits'.
    Returns:
      '__inits'
    """
    # Fast-path already loaded
    if self.__inits is not None:
      return self.__inits
    # Initialize the dictionary
    self.__inits = dict()
    # Populate this dictionary with PyTorch's initialization functions
    for name in dir(torch.nn.init):
      if len(name) == 0 or name[0] == "_": # Ignore "protected" members
        continue
      if name[-1] != "_": # Ignore non-inplace members (heuristic)
        continue
      func = getattr(torch.nn.init, name)
      if isinstance(func, types.FunctionType): # Heuristic
        self.__inits[name[:-1]] = func
    # Return the dictionary
    return self.__inits

  def __init__(self, name_build, config=Configuration(), init_multi=None, init_multi_args=None, init_mono=None, init_mono_args=None, *args, **kwargs):
    """ Model builder constructor.
    Args:
      name_build      Model name or constructor function
      config          Configuration to use for the parameter tensors
      init_multi      Weight initialization algorithm name, or initialization function, for tensors of dimension >= 2
      init_multi_args Additional keyword-arguments for 'init', if 'init' specified as a name
      init_mono       Weight initialization algorithm name, or initialization function, for tensors of dimension == 1
      init_mono_args  Additional keyword-arguments for 'init_mono', if 'init_mono' specified as a name
      ...             Additional (keyword-)arguments forwarded to the constructor
    Notes:
      If possible, data parallelism is enabled automatically
    """
    def make_init(name, args):
      inits = type(self)._get_inits()
      func = inits.get(name, None)
      if func is None:
        raise tools.UnavailableException(inits, name, what="initializer name")
      args = dict() if args is None else args
      def init(params):
        return func(params, **args)
      return init
    # Recover name/constructor
    if callable(name_build):
      name  = tools.fullqual(name_build)
      build = name_build
    else:
      models = type(self)._get_models()
      name  = str(name_build)
      build = models.get(name, None)
      if build is None:
        raise tools.UnavailableException(models, name, what="model name")
    # Recover initialization algorithms
    if isinstance(init_multi, str):
      init_multi = make_init(init_multi, init_multi_args)
    if isinstance(init_mono, str):
      init_mono = make_init(init_mono, init_mono_args)
    # Build model
    with torch.no_grad():
      model = build(*args, **kwargs)
      if not isinstance(model, torch.nn.Module):
        raise tools.UserException(f"Expected built model {name!r} to be an instance of 'torch.nn.Module', found {getattr(type(model), '__name__', '<unknown>')!r} instead")
      # Initialize parameters
      for param in model.parameters():
        if len(param.shape) > 1: # Multi-dimensional
          if init_multi is not None:
            init_multi(param)
        else: # Mono-dimensional
          if init_mono is not None:
            init_mono(param)
      # Move parameters to target device
      model = model.to(**config)
      device = config["device"]
      if device.type == "cuda" and device.index is None: # Model is on GPU and not explicitly restricted to one particular card => enable data parallelism
        model = torch.nn.DataParallel(model)
    params = tools.flatten(model.parameters()) # NOTE: Ordering across runs/nodes seems to be ensured (i.e. only dependent on the model constructor)
    # Finalization
    self._model    = model
    self._name     = name
    self._config   = config
    self._params   = params
    self._gradient = None
    self._defaults = {
      "trainset":  None,
      "testset":   None,
      "loss":      None,
      "criterion": None,
      "optimizer": None }

  def __str__(self):
    """ Compute the "informal", nicely printable string representation of this model.
    Returns:
      Nicely printable string
    """
    return f"model {self._name}"

  @property
  def config(self):
    """ Getter for the immutable configuration.
    Returns:
      Immutable configuration
    """
    return self._config

  def default(self, name, new=None, erase=False):
    """ Get and/or set the named default.
    Args:
      name  Name of the default
      new   Optional new instance, set only if not 'None' or erase is 'True'
      erase Force the replacement by 'None'
    Returns:
      (Old) value of the default
    """
    # Check existence
    if name not in self._defaults:
      raise tools.UnavailableException(self._defaults, name, what="model default")
    # Get current
    old = self._defaults[name]
    # Set if needed
    if erase or new is not None:
      self._defaults[name] = new
    # Return current/old
    return old

  def _resolve_defaults(self, **kwargs):
    """ Resolve the given keyword-arguments with the associated default value.
    Args:
      ... Keyword-arguments, each must have a default if set to None
    Returns:
      In-order given keyword-arguments, with 'None' values replaced with the corresponding default
    """
    res = list()
    for name, value in kwargs.items():
      if value is None:
        value = self.default(name)
        if value is None:
          raise RuntimeError(f"Missing default {name}")
      res.append(value)
    return res

  def run(self, data, training=False):
    """ Run the model at the current parameters for the given input tensor.
    Args:
      data     Input tensor
      training Use training mode instead of testing mode
    Returns:
      Output tensor
    Notes:
      Gradient computation is not enable nor disabled during the run.
    """
    # Set mode
    if training:
      self._model.train()
    else:
      self._model.eval()
    # Compute
    return self._model(data)

  def __call__(self, *args, **kwargs):
    """ Forward call to 'run'.
    Args:
      ... Forwarded (keyword-)arguments
    Returns:
      Forwarded return value
    """
    return self.run(*args, **kwargs)

  def get(self):
    """ Get a reference to the current parameters.
    Returns:
      Flat parameter vector (by reference: future calls to 'set' will modify it)
    """
    return self._params

  def set(self, params, relink=None):
    """ Overwrite the parameters with the given ones.
    Args:
      params Given flat parameter vector
      relink Relink instead of copying (depending on the model, might be faster)
    """
    # Fast path 'set(get())'-like
    if params is self._params:
      return
    # Assignment
    if (self._config.relink if relink is None else relink):
      tools.relink(self._model.parameters(), params)
      self._params = params
    else:
      self._params.copy_(params, non_blocking=self._config["non_blocking"])

  def get_gradient(self):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """
    # Fast path
    if self._gradient is not None:
      return self._gradient
    # Flatten (make if necessary)
    gradient = tools.flatten(tools.grads_of(self._model.parameters()))
    self._gradient = gradient
    return gradient

  def set_gradient(self, gradient, relink=None):
    """ Overwrite the gradient with the given one.
    Args:
      gradient Given flat gradient
      relink   Relink instead of copying (depending on the model, might be faster)
    """
    # Fast path 'set(get())'-like
    if gradient is self._gradient:
      return
    # Assignment
    if (self._config.relink if relink is None else relink):
      tools.relink(tools.grads_of(self._model.parameters()), gradient)
      self._gradient = gradient
    else:
      self.get_gradient().copy_(gradient, non_blocking=self._config["non_blocking"])

  #JS: sampling method for increasing batch size
  def sample_increasing_batch(self, dataset, original_batch_size, current_batch_size, dataset_worker, worker_id=None):
    """ Sample a batch of size current_batch_size from the dataset iterator
    Args:
      dataset                  Training dataset wrapper to use, use the default one if available
      original_batch_size      Batch size at start of learning
      current_batch_size       Desired batch size at the current step
      dataset_worker           Boolean that is true in the case of different dataset iterators for honest workers
      worker_id                Id of the worker (dataset) to sample from (not None in P2P setting)
    Returns:
      A batch of size current_batch_size
    """
    #JS: number of times we have to sample from the iterator to get a batch of size current_batch_size
    numb_samples = int(current_batch_size / original_batch_size)

    if dataset_worker:
        inputs, targets = dataset.sample_worker(worker_id, self._config)
    else:
        inputs, targets = dataset.sample(config=self._config)

    for i in range(1, numb_samples):
        if dataset_worker:
            #JS: have one dataset iterator per honest worker, e.g., in heterogeneous setting or distinct datasets setting for honest workers (e.g., privacy)
            inputs_i, targets_i = dataset.sample_worker(worker_id, self._config)
        else:
            #JS: homogeneous setting, where workers share the dataset
            inputs_i, targets_i = dataset.sample(config=self._config)
        inputs = torch.cat((inputs, inputs_i))
        targets = torch.cat((targets, targets_i))
    return (inputs, targets)

  def loss_localSGD(self, step, other_workers, dataset=None, loss=None, training=None):
    """Estimate loss at the current parameters in local SGD (i.e., only 1 worker in system)"""
    # Recover the defaults, if missing
    dataset, loss = self._resolve_defaults(trainset=dataset, loss=loss)
    # Sample the train batch
    if step == 0:
      inputs, targets = dataset.sample(config=self._config)
    else:
      #sample (nb_workers - 1) batches before
      for i in range(other_workers):
          _, _ = dataset.sample(config=self._config)
      inputs, targets = dataset.sample(config=self._config)

    # Guess whether computation is for training, if necessary
    if training is None:
      training = torch.is_grad_enabled()
    # Forward pass
    return loss(self.run(inputs), targets, self._params), None, None

  #JS: Loss estimation in case of MVR
  def loss(self, labelflipping=False, numb_labels=None, mvr=False, previous_params=None, dataset_worker=False, worker_id=None, batch_increase=None, original_batch_size=None,
    current_batch_size=None, dataset=None, loss=None, training=None):
    """ Estimate honest, previous, and flipped losses at the current parameters, with a batch of the given dataset.
    Args:
      dataset               Training dataset wrapper to use, use the default one if available
      labelflipping         Boolean that is true in case of labelflipping attack
      numb_labels           Number of labels, useful for labelflipping attack
      mvr                   Boolean that is true in case of mvr technique
      previous_params       Previous model parameters in case of mvr technique
      dataset_worker        Boolean that is true in the case of different dataset iterators for workers
      worker_id             Id of the worker (dataset) to sample from
      batch_increase        Boolean that is true when increasing batch size across the learning is enabled
      original_batch_size   Batch size at start of learning (in case of batch increase)
      current_batch_size    Desired batch size at the current step (in case of batch increase)
      loss                  Loss wrapper to use, use the default one if available
      training              Whether this run is for training (instead of testing) purposes, None for guessing (based on whether gradients are computed)
    Returns:
      2 Loss values
    """

    # Recover the defaults, if missing
    dataset, loss = self._resolve_defaults(trainset=dataset, loss=loss)
    # Sample the train batch
    if not batch_increase:
        if dataset_worker:
            #JS: have one dataset iterator per honest worker, e.g., in heterogeneous setting or distinct datasets setting for honest workers (e.g., privacy)
            inputs, targets = dataset.sample_worker(worker_id, self._config)
        else:
            #JS: homogeneous setting, where workers share the dataset
            inputs, targets = dataset.sample(config=self._config)
    else:
        #JS: increasing batch size
        inputs, targets = self.sample_increasing_batch(dataset, original_batch_size, current_batch_size, dataset_worker, worker_id=worker_id)

    # Guess whether computation is for training, if necessary
    if training is None:
      training = torch.is_grad_enabled()

    #JS: Initialize the flipped and previous losses, in case of labelflipping and mvr respectively
    loss_flipped = None
    loss_previous = None

    if labelflipping:
        #JS: Flip the labels for the label flipping attack
        targets_flipped = targets.sub(numb_labels-1).mul(-1)
        loss_flipped = loss(self.run(inputs), targets_flipped, self._params)

    if mvr:
        #JS: Compute the loss at the previous model parameters
        loss_previous = loss(self.run(inputs), targets, previous_params)

    return (loss(self.run(inputs), targets, self._params), loss_flipped, loss_previous)

  #JS: Backpropagation in case of peer to peer model
  @torch.enable_grad()
  def backprop(self, theta=None, labelflipping=False, numb_labels=None, mvr=False, previous_params=None, dataset_worker=False, batch_increase=False, original_batch_size=None,
    current_batch_size=None, worker_id=None, dataset=None, loss=None, outloss=False, step=None, **kwargs):
    """ Estimate gradient at the current parameters, with a batch of the given dataset.
    Args:
      theta                 Parameter vector of the honest worker in question (in case of P2P setting)
      labelflipping         Boolean that is true in case of labelflipping attack
      numb_labels           Number of labels, useful for labelflipping attack
      mvr                   Boolean that is true in case of mvr technique
      previous_params       Previous model parameters in case of mvr technique
      dataset_worker        Boolean that is true in the case of different dataset iterators for workers
      batch_increase        Boolean that is true when increasing batch size across the learning is enabled
      original_batch_size   Batch size at start of learning (in case of batch increase)
      current_batch_size    Desired batch size at the current step (in case of batch increase)
      worker_id             Id of the worker (i.e., dataset) to sample from
      dataset               Training dataset wrapper to use, use the default one if available
      loss                  Loss wrapper to use, use the default one if available
      outloss               Output the loss value as well
      step                  Current time step of the learning
      ...                   Additional keyword-arguments forwarded to 'backprop'
    Returns:
      if 'outloss' is True:
        Tuple of:
        · Flat gradient (by reference: future calls to 'backprop' will modify it)
        · Loss value
      else:
        Flat gradient (by reference: future calls to 'backprop' will modify it)
    """

    if theta is not None:
        #JS: Set the value of the parameter vector to theta (P2P setting)
        self.set(theta)

    #JS: Initalize flipped and previous gradients
    grad_flipped = None
    grad_previous = None

    # Detach and zero the gradient (must be done at each grad to discard computation graph)
    for param in self._params.linked_tensors:
      grad = param.grad
      if grad is not None:
        grad.detach_()
        grad.zero_()

    #JS: Check for local SGD
    if step is None:
        #JS: Typical DSGD
        loss_honest, loss_flipped, loss_previous = self.loss(labelflipping=labelflipping, numb_labels=numb_labels, mvr=mvr, previous_params=previous_params, dataset=dataset, loss=loss,
          dataset_worker=dataset_worker, batch_increase=batch_increase, worker_id=worker_id, original_batch_size=original_batch_size,
          current_batch_size=current_batch_size)
    else:
        #JS: Local SGD
        loss_honest, loss_flipped, loss_previous = self.loss_localSGD(step, 9, dataset=dataset, loss=loss)

    if labelflipping and loss_flipped is not None:
        #JS: Compute the gradient corresponding to the flipped labels
        loss_flipped.backward(**kwargs)
        grad_flipped = copy.deepcopy(self.get_gradient())

        #Detach and zero the gradient again
        for param in self._params.linked_tensors:
            grad = param.grad
            if grad is not None:
                grad.detach_()
                grad.zero_()

    if mvr and loss_previous is not None:
        #JS: Compute the gradient at the previous model parameters
        loss_previous.backward(**kwargs)
        grad_previous = copy.deepcopy(self.get_gradient())

        #Detach and zero the gradient again
        for param in self._params.linked_tensors:
            grad = param.grad
            if grad is not None:
                grad.detach_()
                grad.zero_()

    #JS: Compute the honest/correct gradient (corresponding to the original labels)
    loss_honest.backward(**kwargs)
    grad_honest = self.get_gradient()

    # Relink needed if graph of derivatives was created
    # NOTE: It has been observed that each parameters' grad tensor is a new instance in this case; more investigation may be needed to check whether this relink is really necessary, for now this is a safe behavior
    if "create_graph" in kwargs:
      self._gradient = None
    # Return the flat gradient (and the loss if requested)
    if outloss:
      return (grad_honest, loss_honest, grad_flipped, grad_previous)
    else:
      return (grad_honest, grad_flipped, grad_previous)

  def update(self, gradient, optimizer=None, relink=None):
    """ Update the parameters using the given gradient, and the given optimizer.
    Args:
      gradient  Flat gradient to apply
      optimizer Optimizer wrapper to use, use the default one if available
      relink    Relink instead of copying (depending on the model, might be faster)
    """
    # Recover the defaults, if missing
    optimizer = self._resolve_defaults(optimizer=optimizer)[0]
    # Set the gradient
    self.set_gradient(gradient, relink=(self._config.relink if relink is None else relink))
    # Perform the update step
    optimizer.step()

  @torch.no_grad()
  def eval(self, honest_theta=None, flamby_datasets=None, dataset=None, criterion=None):
    """ Evaluate the model at the current parameters, with a batch of the given dataset.
    Args:
      honest_theta    Parameter vector of the honest worker in question (in case of P2P setting)
      flamby_datasets String that is not None when using Flamby datasets, set to "heart" or "tcg"
      dataset         Testing dataset wrapper to use, use the default one if available
      criterion       Criterion wrapper to use, use the default one if available
    Returns:
      Arithmetic mean of the criterion over the next minibatch
    """

    if honest_theta is not None:
        #JS: Set the value of the parameter vector to honest_theta (in case of P2P setting)
        self.set(honest_theta)

    # Recover the defaults, if missing
    dataset, criterion = self._resolve_defaults(testset=dataset, criterion=criterion)
    # Sample the test batch
    inputs, targets = dataset.sample(flamby_datasets=flamby_datasets, config=self._config)
    # Compute and return the evaluation result
    return criterion(self.run(inputs), targets)
