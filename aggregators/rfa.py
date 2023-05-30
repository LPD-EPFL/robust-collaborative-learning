# coding: utf-8
###
 # @file   RFA.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # RFA (aka Geometric Median) GAR.
 #
 # This algorithm has been introduced in the following paper:
 #   Robust Aggregation for Federated Learning
 #   Krishna Pillutla Sham M. Kakade Zaid Harchaoui
###

from . import register
import torch, math, aggregators

# ---------------------------------------------------------------------------- #
# RFA GAR

def smoothed_weiszfeld(gradients, f, alphas, nu, T, **kwargs):
  """ Smoothed Weiszfeld algorithm
  Args:
    gradients: non-empty list of gradients to aggregate
    f: number of Byzantine gradients
    alphas: scaling factors
    nu: RFA parameter
    T: number of iterations to run the smoothed Weiszfeld algorithm
  Returns:
    Aggregated gradient
  """
  n = len(gradients)
  z = torch.zeros_like(gradients[0])
  #JS: Initialize the algorithm with the coordinate-wise median of gradients
  #z = aggregators.gars.get("median").checked(gradients=gradients, f=f, **kwargs)

  for t in range(T):
    betas = []
    for i in range(n):
      #JS: Compute the norm of z - gradient
      distance = z.sub(gradients[i]).norm().item()
      if math.isnan(distance):
          #JS: distance is infinite
          betas.append(0)
      else:
          betas.append(alphas[i] / max(distance, nu))

    z = torch.zeros_like(gradients[0])
    for grad, beta in zip(gradients, betas):
      if beta != 0:
          z = z.add(grad, alpha=beta)

    z = torch.div(z, sum(betas))
  return z

def aggregate(gradients, f, nu=0.1, T=3, **kwargs):
  """ RFA rule.
  Args:
    gradients: non-empty list of gradients to aggregate
    f: number of Byzantine gradients
    nu: RFA parameter
    T: number of iterations to run the smoothed Weiszfeld algorithm
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  alphas = [1 / len(gradients) for _ in gradients]
  return smoothed_weiszfeld(gradients, f, alphas, nu, T)

def check(gradients, f, nu=0.1, T=3, **kwargs):
  """ Check parameter validity for RFA rule.
  Args:
    gradients: non-empty list of gradients to aggregate
    f: number of Byzantine gradients
    nu: RFA parameter
    T: number of iterations to run the smoothed Weiszfeld algorithm
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  #if not isinstance(f, int) or f < 0 or len(gradients) < 2 * f + 1:
    #return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 0 ≤ f ≤ {(len(gradients) - 1) // 2}."
  if nu < 0:
    return f"Invalid value for nu, got nu = {nu!r}, expected 0 ≤ nu."
  if not isinstance(T, int) or T < 1:
    return f"Invalid value for the number of iterations, got T = {T!r}, expected 1 ≤ T."

def influence(honests, attacks, **kwargs):
  """ Compute the ratio of accepted Byzantine gradients.
  Args:
    honests non-empty list of honest gradients to aggregate
    attacks list of attack gradients to aggregate
    ...     Ignored keyword-arguments
  """
  return len(attacks) / (len(honests) + len(attacks))

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (PyTorch version)
register("rfa", aggregate, check, influence=influence)
