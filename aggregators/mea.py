# coding: utf-8
###
 # @file   mea.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Minimum Eigenvalue Averaging aggregation rule
###

from . import register
import numpy as np
from scipy.linalg import eigh
import torch, itertools


# ---------------------------------------------------------------------------- #

def compute_min_subset(gradients, f, **kwargs):
    """
    gradients:  Non-empty list of gradients to aggregate
    f:          Number of Byzantine gradients to tolerate
    """

    #JS: convert gradients to numpy arrays, and store it in gradients_np
    gradients_np = list()
    for gradient in gradients:
        gradients_np.append(np.array(gradient.cpu()))
    gradients_np = np.array(gradients_np)

    n = len(gradients)
    dimension = gradients_np.shape[1]
    all_subsets = list(itertools.combinations(range(n), n - f))

    min_eigenvalue = None

    for subset in all_subsets:
        subset_gradients = np.take(gradients_np, subset, axis=0)
        subset_gradients_ = subset_gradients.reshape(n-f, 1, dimension)

        avg = np.average(subset_gradients, axis=0)
        cov = np.average(np.array([np.matmul((gradient - avg).T, (gradient - avg)) for gradient in subset_gradients_]), axis=0)
        eig_val, _ = eigh(cov, eigvals=(dimension-1, dimension-1), eigvals_only=False)
        eig_val = eig_val[0]

        if min_eigenvalue is None or min_eigenvalue > eig_val:
            min_eigenvalue = eig_val
            min_subset = subset

    return min_subset

def aggregate(gradients, f, **kwargs):
  """ MEA rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  selected_subset = compute_min_subset(gradients, f, **kwargs)
  selected_gradients = [gradients[j] for j in selected_subset]
  stacked_grads = torch.stack(selected_gradients)
  return stacked_grads.mean(dim=0)

def check(gradients, f, **kwargs):
  """ Check parameter validity for MEA rule.
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
register("mea", aggregate, check)
