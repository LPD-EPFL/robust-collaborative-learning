# coding: utf-8
###
 # @file   anticge.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Attack specifically targeting CGE (see the associated GAR).
###

import math
import torch

from . import register

# ---------------------------------------------------------------------------- #
# Attack implementation

def _nextafter(x, y, d=1e-6):
  """ Return the next floating-point value after x towards y.
  Args:
    x Given floating point
    y Target floating point
    d Some small, positive floating point value
  Returns:
    A floating-point value close to x towards y
  """
  return x * (1. - d) + y * d
nextafter = getattr(math, "nextafter", _nextafter)
del _nextafter

def _compute_normed(grads):
  """ Compute norms and sort gradients by increasing norm, handling non-finite coordinates as belonging to Byzantine gradients.
  Args:
    grads Iterable of gradients
  Returns:
    List of gradients sorted by increasing norm
  """
  def byznorm(grad):
    norm = grad.norm().item()
    return norm if math.isfinite(norm) else math.inf
  return sorted(((byznorm(grad), grad) for grad in grads), key=lambda pair: pair[0])

def attack(grad_honests, f_real, f_decl, **kwargs):
  """ Generate the attack gradients.
  Args:
    grad_honests Non-empty list of honest gradients
    f_real       Number of Byzantine gradients to generate
    f_decl       Number of declared Byzantine gradients
    ...          Ignored keyword-arguments
  Returns:
    Generated Byzantine gradients
  """
  # Trivial attack case: a Byzantine gradient is necessarily selected
  if f_real > f_decl:
    byz_grad = torch.empty_like(grad_honests[0])
    byz_grad.copy_(torch.tensor((math.nan,), dtype=byz_grad.dtype))
    return [byz_grad] * f_real
  # Compute the honest gradient norms, sorted by increasing value
  normed = _compute_normed(grad_honests)
  # Select the maximum norm such that, if Byzantine gradient norms are all strictly below it, every Byzantine gradient is going to be selected
  maxpos = len(grad_honests) - f_decl
  maxnorm = nextafter(normed[maxpos][0], 0)
  # Compute the sum of the would-be selected honest gradients
  attack = normed[0][1].clone().detach_()
  for _, grad in normed[:maxpos]:
    attack.add_(grad)
  # Negate and scale the attack gradient so that it has the maximum norm
  attnorm = attack.norm().item()
  if attnorm > 0:
    attack.mul_(-maxnorm / attnorm)
  # Return same attack gradient for each Byzantine worker
  return [attack] * f_real

def check(grad_honests, f_real, f_decl, **kwargs):
  """ Check parameter validity for this attack template.
  Args:
    grad_honests Non-empty list of honest gradients
    f_real       Number of Byzantine gradients to generate
    ...          Ignored keyword-arguments
  Returns:
    Whether the given parameters are valid for this attack
  """
  if not isinstance(grad_honests, list) or len(grad_honests) == 0:
    return f"Expected a non-empty list of honest gradients, got {grad_honests!r}"
  if not isinstance(f_real, int) or f_real < 0:
    return f"Expected a non-negative number of Byzantine gradients to generate, got {f_real!r}"

# ---------------------------------------------------------------------------- #
# Attack registration

# Register the attack
register("anticge", attack, check)