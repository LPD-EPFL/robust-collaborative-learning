# coding: utf-8
###
 # @file   centeredclip.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Centered clipping GAR.
 #
 # This algorithm has been introduced in the following paper:
 #   Learning from History for Byzantine Robust Optimization.
 #   Sai Praneeth Karimireddy, Lie He, Martin Jaggi
 #   ICML 2021.
###

from . import register
import torch

# ---------------------------------------------------------------------------- #
# Centered clipping GAR

def compute_distances(gradients, f, g_prev, clip_thresh, **kwargs):
  """ Compute (clipped) distances of workers' gradients from previous aggregation result g_prev
  Args:
    gradients: the gradients sent by the workers
    g_prev: previous value of the aggregate gradient
    clip_thresh: clipping threshold
  Returns:
    clipped distance from g_prev
  """
  clipped_distances = list()
  for gradient in gradients:
    distance_vector = gradient.sub(g_prev)
    distance_norm = distance_vector.norm().item()
    if distance_norm > clip_thresh:
      distance_vector.mul_(clip_thresh / distance_norm)
    clipped_distances.append(distance_vector)
  return clipped_distances

def aggregate_step(clipped_distances):
  """ Aggregation step
  Args:
    clipped_distances: list of clipped distances from g_prev
  Returns:
    average of clipped distances
  """
  avg_dist = clipped_distances[0]
  for i in range(1, len(clipped_distances)):
    avg_dist = avg_dist.add(clipped_distances[i])
  return torch.div(avg_dist, len(clipped_distances))

def aggregate(gradients, f, g_prev, clip_thresh, L_iter=3, **kwargs):
  """ Centered Clipping rule.
  Args:
    gradients: Non-empty list of gradients to aggregate
    g_prev: previous value of the gradient
    clip_thresh: clipping threshold
    L_iter: number of iterations to run the clipping/aggregation algorithm
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  for _ in range(L_iter):
    clipped_distances = compute_distances(gradients, f, g_prev, clip_thresh, **kwargs)
    avg_dist = aggregate_step(clipped_distances)
    g_prev = g_prev.add(avg_dist)
  return g_prev

def check(gradients, f, g_prev, clip_thresh, L_iter=3, **kwargs):
  """ Check parameter validity for Centered Clipping rule.
  Args:
    gradients: Non-empty list of gradients to aggregate
    f: Number of Byzantine gradients to tolerate
    clip_thresh: clipping threshold
    L_iter: number of iterations to run the clipping/aggregation algorithm
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  if not isinstance(f, int) or f < 0 or len(gradients) < 2 * f + 1:
    return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f ≤ {(len(gradients) - 1) // 2}"
  if not isinstance(clip_thresh, int) and not isinstance(clip_thresh, float):
    return f"Invalid type for clipping threshold, got type =  {type(clip_thresh)}, expected int/float."
  if clip_thresh < 0:
    return f"Invalid value for clipping threshold, got {clip_thresh}, expected 0 ≤ clip_thresh."

def influence(honests, attacks, f, **kwargs):
  """ Compute the ratio of accepted Byzantine gradients.
  Args:
    honests Non-empty list of honest gradients to aggregate
    attacks List of attack gradients to aggregate
    ...     Ignored keyword-arguments
  """
  return len(attacks) / (len(honests) + len(attacks))

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (PyTorch version)
register("centeredclip", aggregate, check, influence=influence)
