# coding: utf-8
###
 # @file   multiKrum_pseudo.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Pseudo-MultiKrum: GAR that approximates Multi-Krum. Used in the homomorphic setting, due to the expensive nature of homomorphic operations
###

import torch, random
from . import register

# ---------------------------------------------------------------------------- #

def aggregate(gradients, f, k=None, **kwargs):
  """Pseudo-MultiKrum rule: get the k gradients with the smallest pseudo-scores, by running Pseudo Krum k times and averaging the results
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    k         Number of averaged gradients for Pseudo Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  if k is None:
    k = len(gradients) - f

  k_gradients = list()
  #JS: dictionary to hold pairwise distances
  distances = dict()

  #JS: Run Pseudo Krum k times, and store result in list then average
  for _ in range(k):
      #JS: choose (f+1) gradients at random, and compute their pseudo-scores
      indices = range(len(gradients))
      random_indices = random.sample(indices, f+1)

      #JS: compute the pseudo-scores of only these random gradients
      #JS: a pseudo-score is the same as a normal score, but computed only over a random set of (n-f) neighbours
      min_score = min_index = None

      for index in random_indices:
          #JS: gradients[index] is one of the candidates to be outputted by pseudo-Krum
          random_neighbours = random.sample(indices, len(gradients)-f)
          score = 0
          for neighbour in random_neighbours:

              #JS: if index = neighbour, distance = 0 and score is unchanged
              if index == neighbour:
                  continue

              #JS: fetch the distance between gradient and neighbour from dictionary (if found)
              #otherwise calculate it and store it in dictionary
              key = (min(index, neighbour), max(index, neighbour))

              if key in distances:
                  dist = distances[key]
              else:
                  dist = gradients[index].sub(gradients[neighbour]).norm().item()
                  distances[key] = dist

              score += dist**2

          if min_score is None or score < min_score:
              min_score = score
              min_index = index

      k_gradients.append(gradients[min_index])

  stacked_grads = torch.stack(k_gradients)
  return stacked_grads.mean(dim=0)

def check(gradients, f, **kwargs):
  """ Check parameter validity for Pseudo-MultiKrum rule.
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
register("pseudo_multiKrum", aggregate, check)
