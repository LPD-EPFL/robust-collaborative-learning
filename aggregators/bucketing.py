# coding: utf-8
###
 # @file   bucketing.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2022 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Bucketing technique
 #
 # This algorithm has been introduced in the following paper:
 #   Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing
 #   Sai Praneeth Karimireddy, Lie He, Martin Jaggi
 #   ICLR 2022.
###

from . import register
import torch, random, math, aggregators

# ---------------------------------------------------------------------------- #

def aggregate(gradients, f, bucket_size, gar_second, **kwargs):
  """ Bucketing technique.
  Args:
    gradients: Non-empty list of gradients to aggregate
    f: Number of Byzantines to tolerate
    bucket_size: Number of gradients (to be averaged) per bucket
    gar_second: Aggregation rule to aggregate the buckets
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  random.shuffle(gradients)
  number_buckets = math.ceil(len(gradients) / bucket_size)
  buckets=[gradients[i:i + bucket_size] for i in range(0, len(gradients), bucket_size)]
  avg_gradients = list()

  for bucket_id in range(number_buckets):
    stacked_grads = torch.stack(buckets[bucket_id])
    avg_gradients.append(stacked_grads.mean(dim=0))

  defense = aggregators.gars.get(gar_second)
  return defense.checked(gradients=avg_gradients, f=f, **kwargs)

def check(gradients, f, bucket_size, gar_second, **kwargs):
  """ Check parameter validity for Bucketing technique.
  Args:
    gradients: Non-empty list of gradients to aggregate
    f: Number of Byzantines to tolerate
    bucket_size: Number of gradients (to be averaged) per bucket
    gar_second: Aggregation rule to aggregate the buckets
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
  if not isinstance(gar_second, str):
    return f"Invalid type for the gar to use on top of bucketing, got type =  {type(gar_second)}, expected string."

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (PyTorch version)
register("bucketing", aggregate, check)
