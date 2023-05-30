# coding: utf-8
###
 # @file   multikrum.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Multi-Krum aggregation rule
###

import math, torch, tools
from . import register
from itertools import combinations

# ---------------------------------------------------------------------------- #

def compute_distances(gradients, **kwargs):
  """ Compute all pairwise distances between gradients
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    Dictionary of pairwise distances
  """
  distances = dict()
  all_pairs = list(combinations(range(len(gradients)), 2))
  for (x,y) in all_pairs:
      dist = gradients[x].sub(gradients[y]).norm().item()
      if not math.isfinite(dist):
        dist = math.inf
      distances[(x,y)] = dist
  return distances

def get_scores(gradients, f, distances, **kwargs):
  """ Get the scores of gradients sorted increasingly.
  Args:
    gradients   Non-empty list of gradients to aggregate
    f           Number of Byzantine gradients to tolerate
    distances   Dictionary holding all pairwise distances between gradients
    ...       Ignored keyword-arguments
  Returns:
    Gradients with increasing scores
  """

  #JS: compute the scores of gradients
  scores = list()

  for worker_id in range(len(gradients)):
    distances_to_gradient = list()

    #JS: Compute the distances of all other gradients to gradients[worker_id]
    for neighbour in range(len(gradients)):
        if neighbour != worker_id:
            dist = distances.get((min(worker_id, neighbour), max(worker_id, neighbour)), 0)
            distances_to_gradient.append(dist)

    distances_to_gradient.sort()
    score = sum(distances_to_gradient[:len(gradients) - f - 1])
    scores.append((score, worker_id))

  scores.sort(key=lambda x: x[0])
  return scores

def aggregate(gradients, f, k=None, **kwargs):
  """Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    k         Number of averaged gradients for Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  if k is None:
      k = len(gradients) - f
  #JS: Compute all pairwise distances
  distances = compute_distances(gradients)
  #JS: get increasing scores of gradients
  scores = get_scores(gradients, f, distances, **kwargs)
  best_gradients = [gradients[worker_id] for _, worker_id in scores[:k]]
  #JS: return the average of the k gradients with lowest scores
  stacked_grads = torch.stack(best_gradients)
  return stacked_grads.mean(dim=0)

def check(gradients, f, **kwargs):
  """ Check parameter validity for Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  #if not isinstance(f, int) or f < 0 or len(gradients) < 2 * f + 1:
    #return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f ≤ {(len(gradients) - 1) // 2}"

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (Pytorch version)
register("multiKrum", aggregate, check)
