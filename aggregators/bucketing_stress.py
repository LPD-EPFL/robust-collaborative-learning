# coding: utf-8
###
 # @file   bucketing_stress.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2022 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Stress test for Bucketing technique
 #
 # This algorithm has been introduced in the following paper:
 #   Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing
 #   Sai Praneeth Karimireddy, Lie He, Martin Jaggi
 #   ICLR 2022.
###

from . import register
import torch, random, math, aggregators, attacks

# ---------------------------------------------------------------------------- #

def aggregate(gradients, f, bucket_size, gar_second, attack_stress, grads_flipped, z, current_step, mimic_learning_phase, **kwargs):
  """ Bucketing technique.
  Args:
    gradients               Non-empty list of gradients to aggregate
    f                       Number of Byzantines to tolerate
    bucket_size             Number of gradients (to be averaged) per bucket
    gar_second              Aggregation rule to aggregate the buckets
    attack_stress           Attack to be executed by the stress test
    grads_flipped           List of flipped gradients, in case of labelflipping attack
    z                       z in the heuristic of mimic attack
    current_step            Current time step of the learning
    mimic_learning_phase    Number of steps of the learning phase of the mimic heuristic
    ...                     Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """

  random.shuffle(gradients)
  number_buckets = math.ceil(len(gradients) / bucket_size)
  buckets=[gradients[i:i + bucket_size] for i in range(0, len(gradients), bucket_size)]
  honest_buckets = list()

  for bucket_id in range(number_buckets):
    stacked_grads = torch.stack(buckets[bucket_id])
    #JS: compute bucket gradient as the average of the gradients in the bucket
    bucket_gradient = stacked_grads.mean(dim=0)
    #JS: compute norm of bucket gradient, to see the bucket in question is contaminated
    bucket_norm = bucket_gradient.norm().item()
    if not math.isnan(bucket_norm):
        honest_buckets.append(bucket_gradient)

  attack = attacks.attacks.get(attack_stress)
  defense = aggregators.gars.get(gar_second)
  byzantine_buckets = attack.checked(grad_honests=honest_buckets, f_decl=f,
        f_real=f, defense=defense, gar=gar_second, bucket_size=bucket_size,
        grads_flipped=grads_flipped, z=z, current_step=current_step,
        mimic_learning_phase=mimic_learning_phase)
  buckets = honest_buckets + byzantine_buckets
  return defense.checked(gradients=buckets, f=f, **kwargs)

def check(gradients, f, bucket_size, gar_second, attack_stress, grads_flipped, z, current_step, **kwargs):
  """ Check parameter validity for Bucketing technique.
  Args:
    gradients       Non-empty list of gradients to aggregate
    f               Number of Byzantines to tolerate
    bucket_size     Number of gradients (to be averaged) per bucket
    gar_second      Aggregation rule to aggregate the buckets
    attack_stress   Attack to be executed by the stress test
    grads_flipped   List of flipped gradients, in case of labelflipping attack
    z               z in the heuristic of mimic attack
    current_step    Current time step of the learning
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
  if not isinstance(attack_stress, str):
    return f"Invalid type for the attack to use in the stress test, got type =  {type(attack_stress)}, expected string."
  if not isinstance(grads_flipped, list):
    return f"Expected a list of flipped gradients, got {grads_flipped!r}"

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (PyTorch version)
register("bucketing_stress", aggregate, check)
