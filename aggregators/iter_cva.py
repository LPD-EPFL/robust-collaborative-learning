# coding: utf-8
###
 # @file   iter_cva.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2022 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Iterative CVA GAR.
###

import tools
from . import register
import math, torch, aggregators

# ---------------------------------------------------------------------------- #

def compute_closest_gradients(gradients, f, pivot=None, honest_index=None, **kwargs):
  """ Compute the n-f closest gradients to gradients[honest_index] (P2P settting) or pivot(gradients) (PS setting)
  Args:
    gradients       Non-empty list of gradients to aggregate
    f               Number of Byzantine gradients to tolerate
    pivot           Pivot used for CVA. It is a string for an aggregation rule (PS)
    honest_index    Index of the honest worker on which CVA is executed (P2P)
    ...       Ignored keyword-arguments
  Returns:
    List of n-f closest gradients to gradients[honest_index] (P2P) or pivot(gradients) (PS)
  """
  if honest_index is not None:
    #JS: P2P setting
    pivot_gradient = gradients[honest_index]
  elif isinstance(pivot, str):
    #JS: PS setting, i.e., pivot is a string for an aggregation rule
    pivot_gradient = aggregators.gars.get(pivot).checked(gradients=gradients, f=f, **kwargs)
  else:
    #JS: PS setting, i.e., pivot is a gradient
    pivot_gradient = pivot

  gradient_scores = list()
  for i in range(len(gradients)):
    #JS: compute distance to pivot_gradient
    distance = gradients[i].sub(pivot_gradient).norm().item()
    gradient_scores.append((i, distance))
  #JS: sort gradient_scores by increasing distance to pivot_gradient
  gradient_scores.sort(key=lambda x: x[1])
  #JS: return the n-f closest gradients to pivot_gradient
  return [gradients[gradient_scores[j][0]] for j in range(len(gradients) -f)]

def aggregate(gradients, f, numb_iter=2, pivot=None, honest_index=None, **kwargs):
  """ CVA rule.
  Args:
    gradients       Non-empty list of gradients to aggregate
    f               Number of Byzantine gradients to tolerate
    numb_iter       Number of iterations to run the GAR (2 by default)
    pivot           Pivot used for CVA. It is a string for an aggregation rule or a vector (PS)
    honest_index    Index of the honest worker on which CVA is executed (P2P)
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient using Iterative CVA
  """
  #JS: Compute CVA once to get the output gradient
  closest_gradients = compute_closest_gradients(gradients, f, pivot=pivot, honest_index=honest_index, **kwargs)
  stacked_grads = torch.stack(closest_gradients)
  output_grad = stacked_grads.mean(dim=0)
  #JS: Execute CVA numb_iter-1 times and return the result
  for j in range(numb_iter-1):
      closest_gradients = compute_closest_gradients(gradients, f, pivot=output_grad, **kwargs)
      stacked_grads = torch.stack(closest_gradients)
      output_grad = stacked_grads.mean(dim=0)
  return output_grad

def check(gradients, f, pivot=None, honest_index=None, **kwargs):
  """ Check parameter validity for Iterative CVA rule.
  Args:
    gradients       Non-empty list of gradients to aggregate
    f               Number of Byzantine gradients to tolerate
    pivot           Pivot used for CVA. It is a string for an aggregation rule (PS)
    honest_index    Index of the honest worker on which CVA is executed (P2P)
    ...             Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  if not isinstance(honest_index, int) and honest_index is not None:
    return f"Invalid honest index, got type = {type(honest_index)!r}, expected int or None"
  if isinstance(honest_index, int) and (honest_index < 0 or honest_index > (len(gradients)-f-1)):
    return f"Invalid honest index, got honest_index = {honest_index!r}, expected 0 ≤ honest_index ≤ {len(gradients) - f - 1}"
  if not isinstance(pivot, int) and not isinstance(pivot, str) and pivot is not None:
    return f"Invalid honest index, got type = {type(pivot)!r}, expected int or str or None"
  if isinstance(pivot, int) and (pivot < 0 or pivot >= len(gradients)):
    return f"Invalid pivot, got pivot = {pivot!r}, expected 0 ≤ pivot ≤ {len(gradients)-1}"
# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (Pytorch version)
register("iter_cva", aggregate, check)
