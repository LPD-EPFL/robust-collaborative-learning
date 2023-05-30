# coding: utf-8
###
 # @file   cenna.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2022 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # CeNNA technique
 #
 # This algorithm has been designed by us
###

from . import register
import torch, random, math, aggregators

# ---------------------------------------------------------------------------- #

def compute_cva(gradients, f, pivot_gradient):
  """ Compute the average of the n-f closest gradients to pivot_gradient
  Args:
    gradients       Non-empty list of gradients to aggregate
    f               Number of Byzantine gradients to tolerate
    pivot_gradient  Vector that will serve as pivot for CVA
    ...       Ignored keyword-arguments
  Returns:
    Average of the n-f closest gradients to pivot_gradient, i.e., output of CVA on pivot
  """
  gradient_scores = list()
  for i in range(len(gradients)):
    #JS: compute distance to pivot_gradient
    distance = gradients[i].sub(pivot_gradient).norm().item()
    gradient_scores.append((i, distance))
  #JS: sort gradient_scores by increasing distance to pivot_gradient
  gradient_scores.sort(key=lambda x: x[1])
  #JS: Return the average of the n-f closest gradients to pivot_gradient
  closest_gradients = [gradients[gradient_scores[j][0]] for j in range(len(gradients) -f)]
  stacked_grads = torch.stack(closest_gradients)
  return stacked_grads.mean(dim=0)

def aggregate(gradients, f, gar_second, numb_iter=1, **kwargs):
  """ CeNNA technique.
  Args:
    gradients: Non-empty list of gradients to aggregate
    f: Number of Byzantines to tolerate
    gar_second: Aggregation rule to aggregate the buckets
    numb_iter:  Number of iterations to run CeNNA (1 by default)
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  input_gradients = gradients
  for _ in range(numb_iter):
    new_gradients = list()
    for grad in input_gradients:
      #JS: Replace every grad by the output of CVA on input_gradients with grad as pivot
      new_gradients.append(compute_cva(input_gradients, f, grad))
    input_gradients = new_gradients

  defense = aggregators.gars.get(gar_second)
  return defense.checked(gradients=new_gradients, f=f, **kwargs)

def check(gradients, f, gar_second, **kwargs):
  """ Check parameter validity for Bucketing technique.
  Args:
    gradients: Non-empty list of gradients to aggregate
    f: Number of Byzantines to tolerate
    gar_second: Aggregation rule to aggregate the buckets
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  if not isinstance(f, int) or f < 0 or len(gradients) < 2 * f + 1:
    return f"Invalid number of Byzantines to tolerate, got f = {f!r}, expected 0 ≤ f ≤ {(len(gradients) - 1) // 2}"
  if not isinstance(gar_second, str):
    return f"Invalid type for the gar to use on top of bucketing, got type =  {type(gar_second)}, expected string."

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (PyTorch version)
register("cenna", aggregate, check)
