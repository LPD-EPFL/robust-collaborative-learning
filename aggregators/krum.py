# coding: utf-8
###
 # @file   krum_new.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Krum aggregation rule
###

import math
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

def get_gradient_best_score(gradients, f, distances, **kwargs):
  """ Get the gradient with the smallest score.
  Args:
    gradients   Non-empty list of gradients to aggregate
    f           Number of Byzantine gradients to tolerate
    distances   Dictionary holding all pairwise distances between gradients
    ...       Ignored keyword-arguments
  Returns:
    Gradient with smallest score
  """

  #JS: compute the scores of gradients, and return the gradient with smallest score
  min_score = min_index = None

  for worker_id in range(len(gradients)):
    distances_to_gradient = list()

    #JS: Compute the distances of all other gradients to gradients[worker_id]
    for neighbour in range(len(gradients)):
        if neighbour != worker_id:
            dist = distances.get((min(worker_id, neighbour), max(worker_id, neighbour)), 0)
            distances_to_gradient.append(dist)

    distances_to_gradient.sort()
    score = sum(distances_to_gradient[:len(gradients) - f - 1])

    #JS: update min score
    if min_score is None or score < min_score:
      min_score = score
      min_index = worker_id

  return gradients[min_index]

def aggregate(gradients, f, **kwargs):
  """Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  # Compute all pairwise distances
  distances = compute_distances(gradients)
  return get_gradient_best_score(gradients, f, distances, **kwargs)

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
register("krum", aggregate, check)
