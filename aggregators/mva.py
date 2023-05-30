# coding: utf-8
###
 # @file   mva.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2022 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Minimum Variance Averaging (MVA) GAR.
###

import tools
from . import register
import itertools, torch

# ---------------------------------------------------------------------------- #
# MVA GAR

def compute_min_subset(gradients, f, **kwargs):
  """ Compute the subset of minimum variance.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Set of indices corresponding to "optimal" subset
  """
  n = len(gradients)
  all_subsets = list(itertools.combinations(range(n), n - f))

  min_variance = None

  for subset in all_subsets:
      subset_gradients = [gradients[j] for j in subset]
      stacked_grads = torch.stack(subset_gradients)
      avg_gradient = stacked_grads.mean(dim=0)

      current_variance = 0
      for gradient in subset_gradients:
          distance_from_average = gradient.sub(avg_gradient).norm().item()
          current_variance += distance_from_average**2

      if min_variance is None or min_variance > current_variance:
          min_variance = current_variance
          min_subset = subset

  return min_subset

def aggregate(gradients, f, **kwargs):
  """ MVA rule.
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
  """ Check parameter validity for MVA rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  #if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 1:
    #return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(len(gradients) - 1) // 2}"

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (Pytorch version)
register("mva", aggregate, check)
