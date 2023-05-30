# coding: utf-8
###
 # @file   filterL2.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # FilterL2 aggregation rule
 # https://www.stat.berkeley.edu/~jsteinhardt/publications/thesis/paper.pdf
###

from . import register
import numpy as np
from scipy.linalg import eigh
import torch

# ---------------------------------------------------------------------------- #

def filterL2(gradients, f, sigma, expansion=20, **kwargs):
    """
    gradients:  Non-empty list of gradients to aggregate
    f:          Number of Byzantine gradients to tolerate
    sigma:      operator norm of covariance matrix assumption
    """

    #JS: handle exception when sigma = 0
    if sigma == 0:
        sigma = 0.00001

    #JS: convert gradients to numpy arrays, and store it in gradients_np
    gradients_np = list()
    for gradient in gradients:
        gradients_np.append(np.array(gradient.cpu()))
    gradients_np = np.array(gradients_np)

    n = gradients_np.shape[0]
    dimension = gradients_np.shape[1]
    gradients_ = gradients_np.reshape(n, 1, dimension)

    c = np.ones(n)
    for i in range(2 * f):
        avg = np.average(gradients_np, axis=0, weights=c)
        cov = np.average(np.array([np.matmul((gradient - avg).T, (gradient - avg)) for gradient in gradients_]), axis=0, weights=c)
        eig_val, eig_vec = eigh(cov, eigvals=(dimension-1, dimension-1), eigvals_only=False)
        eig_val = eig_val[0]
        eig_vec = eig_vec.T[0]

        if eig_val**2 <= expansion * sigma**2:
            return torch.from_numpy(avg)

        tau = np.array([np.inner(gradient-avg, eig_vec)**2 for gradient in gradients_np])
        tau_max_idx = np.argmax(tau)
        tau_max = tau[tau_max_idx]
        c = c * (1 - tau/tau_max)

        gradients_np = np.concatenate((gradients_np[:tau_max_idx], gradients_np[tau_max_idx+1:]))
        gradients_ = gradients_np.reshape(-1, 1, dimension)
        c = np.concatenate((c[:tau_max_idx], c[tau_max_idx+1:]))
        c = c / np.linalg.norm(c, ord=1)

    avg = np.average(gradients_np, axis=0, weights=c)
    return torch.from_numpy(avg)

def aggregate(gradients, f, sigma_filterL2, device, **kwargs):
  """FilterL2 rule.
  Args:
    gradients       Non-empty list of gradients to aggregate
    f               Number of Byzantine gradients to tolerate
    sigma_filterL2: Sigma parameter for FilterL2
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  return filterL2(gradients, f, sigma_filterL2, **kwargs).to(device=device)

def check(gradients, f, **kwargs):
  """ Check parameter validity for FilterL2 rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (Pytorch version)
register("filterL2", aggregate, check)
