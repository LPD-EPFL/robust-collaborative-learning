# coding: utf-8
###
 # @file   krum_pseudo.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Pseudo-Krum: GAR that approximates Krum. Used in the homomorphic setting, due to the expensive nature of homomorphic operations
###

from . import register
import random

# ---------------------------------------------------------------------------- #

def get_gradient_best_score(gradients, f, **kwargs):
  """ Get the gradient with the smallest pseudo-score.
  Args:
    gradients   Non-empty list of gradients to aggregate
    f           Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Gradient with smallest pseudo-score
  """

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
          #JS: compute distance between gradient and neighbour, and add the square of it to the score
          dist = gradients[index].sub(gradients[neighbour]).norm().item()
          score += dist**2
      if min_score is None or score < min_score:
          min_score = score
          min_index = index

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
  return get_gradient_best_score(gradients, f, **kwargs)

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

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (Pytorch version)
register("pseudoKrum", aggregate, check)
