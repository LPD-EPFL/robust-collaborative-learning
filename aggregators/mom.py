# coding: utf-8
###
 # @file   mom.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2022 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Median of Means (MOM) technique
 #
###

from . import register
import torch, random, math, aggregators

# ---------------------------------------------------------------------------- #

def aggregate(gradients, f, bucket_size, **kwargs):
  """ MOM technique.
  Args:
    gradients: Non-empty list of gradients to aggregate
    f: Number of Byzantines to tolerate
    bucket_size: Number of gradients (to be averaged) per bucket
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  random.Random(4).shuffle(gradients)
  number_buckets = math.ceil(len(gradients) / bucket_size)
  buckets=[gradients[i:i + bucket_size] for i in range(0, len(gradients), bucket_size)]
  avg_gradients = list()

  for bucket_id in range(number_buckets):
    stacked_grads = torch.stack(buckets[bucket_id])
    avg_gradients.append(stacked_grads.mean(dim=0))

  defense = aggregators.gars.get("rfa")
  return defense.checked(gradients=avg_gradients, f=f, **kwargs)

def check(gradients, f, bucket_size, **kwargs):
  """ Check parameter validity for MOM technique.
  Args:
    gradients: Non-empty list of gradients to aggregate
    f: Number of Byzantines to tolerate
    bucket_size: Number of gradients (to be averaged) per bucket
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  if not isinstance(f, int) or f < 0 or len(gradients) < 4 * f + 1:
    return f"Invalid number of Byzantines to tolerate, got f = {f!r}, expected 0 ≤ f ≤ {(len(gradients) - 1) // 4}"
  if not isinstance(bucket_size, int) or bucket_size < 2 or bucket_size > len(gradients):
    return f"Invalid bucket size. Should be an integer between 1 and {len(gradients)}"

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (PyTorch version)
register("mom", aggregate, check)
