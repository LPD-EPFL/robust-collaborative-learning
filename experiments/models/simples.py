# coding: utf-8
###
 # @file   simples.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Collection of simple models.
###

__all__ = ["full", "small", "fullTan", "little", "conv", "logit", "logitm", "linear", "jaggi"]

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------- #
# Simple fully-connected model, for MNIST
class _Full(torch.nn.Module):
	""" Simple, small fully connected model.
	"""

	def __init__(self):
		""" Model parameter constructor.
		"""
		super().__init__()
		# Build parameters
		self._f1 = torch.nn.Linear(28 * 28, 100)
		self._f2 = torch.nn.Linear(100, 10)

	def forward(self, x):
		""" Model's forward pass.
		Args:
			x Input tensor
		Returns:
			Output tensor
		"""
		# Forward pass
		x = torch.nn.functional.relu(self._f1(x.view(-1, 28 * 28)))
		x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
		return x

def full(*args, **kwargs):
	""" Build a new simple, fully connected model.
	Args:
		... Forwarded (keyword-)arguments
	Returns:
		Fully connected model
	"""
	global _Full
	return _Full(*args, **kwargs)


#JS Smaller simple fully-connected model, for MNIST
class _Small(torch.nn.Module):

	def __init__(self):
		""" Model parameter constructor.
		"""
		super().__init__()
		# Build parameters
		self._f1 = torch.nn.Linear(28 * 28, 50)
		self._f2 = torch.nn.Linear(50, 10)

	def forward(self, x):
		""" Model's forward pass.
		Args:
			x Input tensor
		Returns:
			Output tensor
		"""
		# Forward pass
		x = torch.nn.functional.relu(self._f1(x.view(-1, 28 * 28)))
		x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
		return x

def small(*args, **kwargs):
	""" Build a new simple, fully connected model.
	Args:
		... Forwarded (keyword-)arguments
	Returns:
		Fully connected model
	"""
	global _Small
	return _Small(*args, **kwargs)


# Simple fully-connected model for MNIST, with tanh instead of relu
class _FullTan(torch.nn.Module):
	""" Simple, small fully connected model.
	"""

	def __init__(self):
		""" Model parameter constructor.
		"""
		super().__init__()
		# Build parameters
		self._f1 = torch.nn.Linear(28 * 28, 100)
		self._f2 = torch.nn.Linear(100, 10)

	def forward(self, x):
		""" Model's forward pass.
		Args:
			x Input tensor
		Returns:
			Output tensor
		"""
		# Forward pass
		x = torch.tanh(self._f1(x.view(-1, 28 * 28)))
		x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
		return x

def fullTan(*args, **kwargs):
	""" Build a new simple, fully connected model.
	Args:
		... Forwarded (keyword-)arguments
	Returns:
		Fully connected model
	"""
	global _FullTan
	return _FullTan(*args, **kwargs)

# ---------------------------------------------------------------------------- #
#JS: Simple fully-connected model for MNIST, exactly as stated in ICLR paper
class _Little(torch.nn.Module):
	""" Simple, small fully connected model.
	"""

	def __init__(self):
		""" Model parameter constructor.
		"""
		super().__init__()
		# Build parameters
		self._f1 = torch.nn.Linear(28 * 28, 100)
		self._f2 = torch.nn.Linear(100, 10)

	def forward(self, x):
		""" Model's forward pass.
		Args:
			x Input tensor
		Returns:
			Output tensor
		"""
		# Forward pass
		x = torch.nn.functional.relu(self._f1(x.view(-1, 28 * 28)))
		x = torch.nn.functional.log_softmax(torch.nn.functional.relu(self._f2(x)), dim=1)
		return x

def little(*args, **kwargs):
	""" Build a new simple, fully connected model.
	Args:
		... Forwarded (keyword-)arguments
	Returns:
		Fully connected model
	"""
	global _Little
	return _Little(*args, **kwargs)


# ---------------------------------------------------------------------------- #
# Simple convolutional model, for MNIST

class _Conv(torch.nn.Module):
	""" Simple, small convolutional model."""

	def __init__(self):
		""" Model parameter constructor.
		"""
		super().__init__()
		# Build parameters
		self._c1 = torch.nn.Conv2d(1, 20, 5, 1)
		self._c2 = torch.nn.Conv2d(20, 50, 5, 1)
		self._f1 = torch.nn.Linear(800, 500)
		self._f2 = torch.nn.Linear(500, 10)

	def forward(self, x):
		""" Model's forward pass.
		Args:
			x Input tensor
		Returns:
			Output tensor
		"""
		# Forward pass
		x = torch.nn.functional.relu(self._c1(x))
		x = torch.nn.functional.max_pool2d(x, 2, 2)
		x = torch.nn.functional.relu(self._c2(x))
		x = torch.nn.functional.max_pool2d(x, 2, 2)
		x = torch.nn.functional.relu(self._f1(x.view(-1, 800)))
		x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
		return x

def conv(*args, **kwargs):
	""" Build a new simple, convolutional model.
	Args:
		... Forwarded (keyword-)arguments
	Returns:
		Convolutional model
	"""
	global _Conv
	return _Conv(*args, **kwargs)


#JS: CNN model for MNIST used in bucketing paper
class _Jaggi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def jaggi(*args, **kwargs):
	""" Build a CNN model from bucketing paper.
	Args:
		... Forwarded (keyword-)arguments
	Returns:
		CNN model
	"""
	global _Jaggi
	return _Jaggi(*args, **kwargs)

# ---------------------------------------------------------------------------- #
# Simple logistic regression model (for phishing)

class _Logit(torch.nn.Module):
	""" Simple logistic regression model.
	"""

	def __init__(self, din, dout=1):
		""" Model parameter constructor.
		Args:
			din  Number of input dimensions
			dout Number of output dimensions
		"""
		super().__init__()
		# Store model parameters
		self._din  = din
		self._dout = dout
		# Build parameters
		self._linear = torch.nn.Linear(din, dout)

	def forward(self, x):
		""" Model's forward pass.
		Args:
			x Input tensor
		Returns:
			Output tensor
		"""
		return torch.sigmoid(self._linear(x.view(-1, self._din)))

def logit(*args, **kwargs):
	""" Build a new logistic regression model.
	Args:
		... Forwarded (keyword-)arguments
	Returns:
		Logistic regression model
	"""
	global _Logit
	return _Logit(*args, **kwargs)

# ---------------------------------------------------------------------------- #
# Simple logistic regression model for MNIST

class _Logitm(torch.nn.Module):
	""" Simple logistic regression model.
	"""

	def __init__(self):
		""" Model parameter constructor.
		Args:
			din  Number of input dimensions
			dout Number of output dimensions
		"""
		super().__init__()
		# Build parameters
		self._linear = torch.nn.Linear(784, 10)

	def forward(self, x):
		""" Model's forward pass.
		Args:
			x Input tensor
		Returns:
			Output tensor
		"""
		return torch.sigmoid(self._linear(x.view(-1, 784)))

def logitm(*args, **kwargs):
	""" Build a new logistic regression model.
	Args:
		... Forwarded (keyword-)arguments
	Returns:
		Logistic regression
	"""
	global _Logitm
	return _Logitm(*args, **kwargs)

# ---------------------------------------------------------------------------- #
# Simple(st) linear model

class _Linear(torch.nn.Module):
	""" Simple linear model.
	"""

	def __init__(self, din, dout=1):
		""" Model parameter constructor.
		Args:
			din  Number of input dimensions
			dout Number of output dimensions
		"""
		super().__init__()
		# Store model parameters
		self._din  = din
		self._dout = dout
		# Build parameters
		self._linear = torch.nn.Linear(din, dout)

	def forward(self, x):
		""" Model's forward pass.
		Args:
			x Input tensor
		Returns:
			Output tensor
		"""
		return self._linear(x.view(-1, self._din))

def linear(*args, **kwargs):
	""" Build a new linear model.
	Args:
		... Forwarded (keyword-)arguments
	Returns:
		Linear model
	"""
	global _Linear
	return _Linear(*args, **kwargs)
