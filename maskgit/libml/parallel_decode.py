# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fast decoding routines for non-autoregressive generation."""

import flax
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from maskgit.libml import mask_schedule

# Confidence score for known tokens to avoid masking or repredicting them.
# Here we don't use 1.0 because the upper bounder of the probability can be
# possiblity larger than 1 due to the noise addition.
_CONFIDENCE_OF_KNOWN_TOKENS = jnp.inf


def mask_by_random_topk(rng, mask_len, probs, temperature=1.0):
  """Modifies from jax.random.choice without replacement.

  JAX's original implementation is as below:
    g = -gumbel(key, (n_inputs,)) - jnp.log(p)
    ind = jnp.argsort(g)[:n_draws]
  We adds temperature annealing on top of it, which is:
    g = -gumbel(key, (n_inputs,)) - temperature * jnp.log(p)
    ind = jnp.argsort(g)[:n_draws]

  Args:
    rng: a PRNG key used as the random key.
    mask_len: the number to mask.
    probs: the probabilities associated with each entry.
    temperature: when temperature = 1.0, it's identical to jax's implementation.
      The larger this value is, the more random the masking is picked.

  Returns:
    A binary masking map [batch_size, seq_len].
  """
  confidence = jnp.log(probs) + temperature * jax.random.gumbel(
      rng, probs.shape)
  sorted_confidence = jnp.sort(confidence, axis=-1)
  # Obtains cut off threshold given the mask lengths.
  cut_off = jnp.take_along_axis(sorted_confidence, mask_len.astype(jnp.int32), axis=-1)
  # Masks tokens with lower confidence.
  masking = (confidence < cut_off)
  return masking


@flax.struct.dataclass
class State:
  """Holds decoding state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jnp.DeviceArray  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  cur_seqs: jnp.DeviceArray  # int32 [batch, seq_len]
  rng: jnp.DeviceArray  # Sampling random state.
  final_seqs: jnp.DeviceArray  # int32 [batch, num_iter, seq_len]

@flax.struct.dataclass
class StateWithLogits(State):
  logits: jnp.DeviceArray  # float32 [batch, seq_len, vocab_size]

def state_init(init_indices, rng, num_iter, start_iter=0):
  """Initializes the decoding state data structure."""
  cur_index0 = jnp.array(start_iter)
  cur_seqs0 = init_indices
  final_seqs0 = jnp.expand_dims(init_indices, 1)
  final_seqs0 = jnp.tile(final_seqs0, (1, num_iter, 1))
  return State(
      cur_index=cur_index0, cur_seqs=cur_seqs0, rng=rng, final_seqs=final_seqs0)

def state_init_with_logits(init_indices, rng, num_iter, codebook_size, start_iter=0):
  """Initializes the decoding state data structure."""
  cur_index0 = jnp.array(start_iter)
  cur_seqs0 = init_indices
  final_seqs0 = jnp.expand_dims(init_indices, 1)
  final_seqs0 = jnp.tile(final_seqs0, (1, num_iter, 1))
  logits = jnp.zeros((init_indices.shape[0], init_indices.shape[1], codebook_size))
  return StateWithLogits(
      cur_index=cur_index0, cur_seqs=cur_seqs0, rng=rng, final_seqs=final_seqs0, logits=logits)


def decode(inputs,
           rng,
           tokens_to_logits,
           mask_token_id=-1,
           num_iter=12,
           start_iter=0,
           choice_temperature=1.0,
           mask_scheduling_method="cosine"):
  """Fast decoding for iterative generation.

  Args:
    inputs: int32 array: [batch_size, seq_length] input sequence of masked
      tokens, where the masking tokens is defined by mask_token_id.
    rng: jnp.DeviceArray: sampling random state.
    tokens_to_logits: decoder function taking single token slices and cache and
      returning logits and updated cache.
    mask_token_id: int: [Mask] token id.
    num_iter: int: default is 12.
    start_iter: int: default is 0.
    choice_temperature: float: temperature to control the randomness of masking.
    mask_scheduling_method: masking method string. See mask_schedule.py for
      details.

  Returns:
     [batch_size, num_iter, seq_length] output sequence of tokens in all
       iterations.
  """
  inputs = inputs.astype("int32")
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  # Initializes state
  init_state = state_init(inputs, rng, num_iter, start_iter=start_iter)

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < num_iter)
    return not_at_end

  def loop_body_fn(state):
    """Beam search loop state update function.
      JY notes:
      - samples and fills in for all masked tokens.
      - identifies masked tokens for next round according to sample confidence.
    """
    rng = state.rng
    step = state.cur_index
    # Current input ids: [batch_size, seq_length].
    cur_ids = state.cur_seqs

    # Calls model on current seqs to get next-iteration seqs.
    logits = tokens_to_logits(cur_ids)
    rng, sample_rng = jax.random.split(rng, 2)
    # Samples the ids using categorical sampling: [batch_size, seq_length].
    sampled_ids = jax.random.categorical(sample_rng, logits)

    # Just updates the masked tokens.
    unknown_map = (cur_ids == mask_token_id)
    sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
    # Defines the mask ratio for the next round. The number to mask out is
    # determined by mask_ratio * unknown_number_in_the_beginning.
    ratio = 1. * (step + 1) / num_iter
    mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                        mask_scheduling_method)
    # Updates final seqs with the current sampled_ids.
    final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
    # Computes the probabilities of each selected tokens.
    probs = jax.nn.softmax(logits, axis=-1)
    selected_probs = jnp.squeeze(
        jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids.astype(jnp.int32), -1), -1), -1)
    # Ignores the tokens given in the input by overwriting their confidence.
    selected_probs = jnp.where(unknown_map, selected_probs,
                               _CONFIDENCE_OF_KNOWN_TOKENS)
    # Gets mask lens for each sample in the batch according to the mask ratio.
    mask_len = jnp.expand_dims(
        jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
    # Keeps at least one of prediction in this round and also masks out at least
    # one and for the next iteration
    mask_len = jnp.maximum(
        1,
        jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

    # Adds noise for randomness
    rng, choice_rng = jax.random.split(rng)
    masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                  choice_temperature * (1. - ratio))
    # Masks tokens with lower confidence.
    sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)
    return State(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        rng=rng,
        final_seqs=final_seqs)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  return final_state.final_seqs

def rescale_confidence(confidence: float, temp_min: float = 1e-6, temp_max: float = 100):
  """Rescales confidence from 0-1 to a more drastic scale logarithmically"""
  return jnp.exp(jnp.log(temp_min) + jnp.minimum(jnp.maximum(confidence, 0), 1) * (jnp.log(temp_max) - jnp.log(temp_min)))

def guidance_to_reweight(reweight_kernel: jnp.ndarray, guidance: jnp.ndarray, confidence: jnp.ndarray):
  r"""
    reweight_kernel: C x C
    guidance: B S, first token for cls dropped
    confidence: B S, ^

    return: B S C
  """
  guidance = reweight_kernel[guidance[:, 1:]]
  reweight = jax.nn.softmax(guidance * rescale_confidence(confidence[:, 1:, None]), axis=-1)
  reweight = reweight / jnp.mean(reweight, axis=-1, keepdims=True) # This will blow up if delta distribution, but that should only for extreme confidence, which never happens AFAIC
  reweight = jnp.concatenate([jnp.ones([reweight.shape[0], 1, reweight.shape[-1]]), reweight], axis=1)
  return reweight

def decode_self_guidance(inputs,
           guidance, # unmasked, [batch_size, seq_length]
           rng,
           tokens_to_logits,
           mask_token_id=-1,
           num_iter=12,
           start_iter=0,
           choice_temperature=1.0,
           mask_scheduling_method="cosine",
           self_guidance=None, # Code x Code matrix. No need for normalization, an energy is fine.
           confidence=1., # exponential in mixing
  ):
  inputs = inputs.astype("int32")
  guidance = guidance.astype("int32")
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  # Initializes state
  init_state = state_init(inputs, rng, num_iter, start_iter=start_iter)

  if self_guidance is None:
    self_guidance_weighted = jnp.ones((guidance.shape[0], guidance.shape[0]), dtype=jnp.float32)
    self_guidance_weighted = jnp.concatenate([jnp.ones_like(self_guidance_weighted[:,:1]), self_guidance_weighted], axis=1)
  else:
    self_guidance_weighted = guidance_to_reweight(self_guidance, guidance, jnp.full_like(guidance, confidence))

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < num_iter)
    return not_at_end

  def loop_body_fn(state):
    """Beam search loop state update function.
      JY notes:
      - samples and fills in for all masked tokens.
      - identifies masked tokens for next round according to sample confidence.
    """
    rng = state.rng
    step = state.cur_index
    # Current input ids: [batch_size, seq_length].
    cur_ids = state.cur_seqs

    # Calls model on current seqs to get next-iteration seqs.
    logits = tokens_to_logits(cur_ids) # B T Code_tgt
    # Just updates the masked tokens.
    unknown_map = (cur_ids == mask_token_id)
    self_guidance_weighted = guidance_to_reweight(self_guidance, guidance, jnp.full_like(guidance, confidence))
    self_guidance_weighted = jnp.where(unknown_map[..., None], self_guidance_weighted, 1.)
    # * Pretty sure we don't ever re-mask a known token, so this should be fine i.e. but we regenerate for convenience (rather than calling global)
    logits = self_guidance_weighted * logits # * Change

    rng, sample_rng = jax.random.split(rng, 2)
    # Samples the ids using categorical sampling: [batch_size, seq_length].
    sampled_ids = jax.random.categorical(sample_rng, logits)

    sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
    # Defines the mask ratio for the next round. The number to mask out is
    # determined by mask_ratio * unknown_number_in_the_beginning.
    ratio = 1. * (step + 1) / num_iter
    mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                        mask_scheduling_method)
    # Updates final seqs with the current sampled_ids.
    final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
    # Computes the probabilities of each selected tokens.
    probs = jax.nn.softmax(logits, axis=-1)
    selected_probs = jnp.squeeze(
        jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids.astype(jnp.int32), -1), -1), -1)
    # Ignores the tokens given in the input by overwriting their confidence.
    selected_probs = jnp.where(unknown_map, selected_probs,
                               _CONFIDENCE_OF_KNOWN_TOKENS)
    # Gets mask lens for each sample in the batch according to the mask ratio.
    mask_len = jnp.expand_dims(
        jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
    # Keeps at least one of prediction in this round and also masks out at least
    # one and for the next iteration
    mask_len = jnp.maximum(
        1,
        jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

    # Adds noise for randomness
    rng, choice_rng = jax.random.split(rng)
    masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                  choice_temperature * (1. - ratio))
    # Masks tokens with lower confidence.
    sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)
    return State(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        rng=rng,
        final_seqs=final_seqs)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  return final_state.final_seqs

def decode_self_guidance_iterate(inputs,
           guidance, # unmasked, [batch_size, seq_length]
           rng,
           tokens_to_logits,
           mask_token_id=-1,
           num_iter=12,
           start_iter=0,
           choice_temperature=1.0,
           mask_scheduling_method="cosine",
           self_guidance=None, # Code x Code matrix. No need for normalization, an energy is fine.
           confidence=1., # exponential in mixing
  ):
  r"""
    Modification where `tokens_to_logits` expects to receive guidance_ids, and it directly integrates (see `iterate_guidance` case in `flax_manual`)
  """
  inputs = inputs.astype("int32")
  guidance = guidance.astype("int32")
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  # Initializes state
  init_state = state_init(inputs, rng, num_iter, start_iter=start_iter)

  if self_guidance is None:
    self_guidance_weighted = jnp.ones((guidance.shape[0], guidance.shape[0]), dtype=jnp.float32)
    self_guidance_weighted = jnp.concatenate([jnp.ones_like(self_guidance_weighted[:,:1]), self_guidance_weighted], axis=1)
  else:
    self_guidance_weighted = guidance_to_reweight(self_guidance, guidance, jnp.full_like(guidance, confidence))

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < num_iter)
    return not_at_end

  def loop_body_fn(state):
    """Beam search loop state update function.
      JY notes:
      - samples and fills in for all masked tokens.
      - identifies masked tokens for next round according to sample confidence.
    """
    rng = state.rng
    step = state.cur_index
    # Current input ids: [batch_size, seq_length].
    cur_ids = state.cur_seqs
    unknown_map = (cur_ids == mask_token_id)

    # Calls model on current seqs to get next-iteration seqs.
    masked_guidance = jnp.where(unknown_map, guidance, mask_token_id)
    logits = tokens_to_logits(cur_ids, masked_guidance) # B T Code_tgt
    # Just updates the masked tokens.
    self_guidance_weighted = guidance_to_reweight(self_guidance, guidance, jnp.full_like(guidance, confidence))
    self_guidance_weighted = jnp.where(unknown_map[..., None], self_guidance_weighted, 1.)
    # * Pretty sure we don't ever re-mask a known token, so this should be fine i.e. but we regenerate for convenience (rather than calling global)
    logits = self_guidance_weighted * logits # * Change

    rng, sample_rng = jax.random.split(rng, 2)
    # Samples the ids using categorical sampling: [batch_size, seq_length].
    sampled_ids = jax.random.categorical(sample_rng, logits)

    sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
    # Defines the mask ratio for the next round. The number to mask out is
    # determined by mask_ratio * unknown_number_in_the_beginning.
    ratio = 1. * (step + 1) / num_iter
    mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                        mask_scheduling_method)
    # Updates final seqs with the current sampled_ids.
    final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
    # Computes the probabilities of each selected tokens.
    probs = jax.nn.softmax(logits, axis=-1)
    selected_probs = jnp.squeeze(
        jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids.astype(jnp.int32), -1), -1), -1)
    # Ignores the tokens given in the input by overwriting their confidence.
    selected_probs = jnp.where(unknown_map, selected_probs,
                               _CONFIDENCE_OF_KNOWN_TOKENS)
    # Gets mask lens for each sample in the batch according to the mask ratio.
    mask_len = jnp.expand_dims(
        jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
    # Keeps at least one of prediction in this round and also masks out at least
    # one and for the next iteration
    mask_len = jnp.maximum(
        1,
        jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

    # Adds noise for randomness
    rng, choice_rng = jax.random.split(rng)
    masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                  choice_temperature * (1. - ratio))
    # Masks tokens with lower confidence.
    sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)
    return State(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        rng=rng,
        final_seqs=final_seqs)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  return final_state.final_seqs

def decode_context_guidance(inputs,
           guidance, # unmasked
           rng,
           tokens_to_logits,
           codebook_size=8192,
           mask_token_id=-1,
           num_iter=12,
           start_iter=0,
           choice_temperature=1.0,
           mask_scheduling_method="cosine",
  ):
  """
  We want to decode while conditioning on user guidance.
  We still use the schedule, but use it to determine a priori how many tokens to mask and keep the rest of input as context.

  Self-guidance not implemented here because pilots show dependence here is too strong (for untuned model).
  """
  inputs = inputs.astype("int32")
  guidance = guidance.astype("int32")
  breakpoint() # We changed guidance to not include the class token, how does that affect things? we probably want to pad by one to match input shape
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  # Initializes state
  init_state = state_init_with_logits(inputs, rng, num_iter, codebook_size=codebook_size, start_iter=start_iter)

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < num_iter)
    return not_at_end

  def loop_body_fn(state):
    """Beam search loop state update function. Inverts order of sampling and masking.
      JY notes:
      - we still keep post-masking as before. i.e. loop comes masked, sample, and mask again.
        - But of the masked tokens, we replace a fraction with guidance.
        - This fraction is determined by previous round confidence.
      - Specifically we will highlight a small number of tokens to mask
        - (either from uniform or according to same confidence logic as above, using confidence from previous round),
        - the rest are guided (for this round).

      Hmm... the order of ops is not right -- if we replace masks with guidance then there's a shift,
      i.e. we shouldn't believe the confidences from those guided tokens to be calibrated for subsequent sampling.
      Oh well. BERT does it. We can too.
    """
    rng = state.rng
    step = state.cur_index
    rng, sample_rng = jax.random.split(rng, 2)
    # Current input ids: [batch_size, seq_length].
    cur_ids = state.cur_seqs
    unknown_map = (cur_ids == mask_token_id)

    # Move mask calculations upfront, so we can decide how much guidance to use.
    # Mask length determines the number of tokens we want masked in the end, and thus the number we replace with guidance.

    ratio = 1. * step / num_iter
    mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                        mask_scheduling_method)

    # Gets mask lens for each sample in the batch according to the mask ratio.
    mask_len = jnp.expand_dims(
        jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
    # Keeps at least one of prediction in this round and also masks out at least
    # one and for the next iteration
    mask_len = jnp.maximum(
        1,
        jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

    # Computes the probabilities of each selected tokens.
    if state.logits is None:
      selected_probs = jnp.ones(cur_ids.shape[0], cur_ids.shape[1]) / codebook_size
    else:
      probs = jax.nn.softmax(state.logits, axis=-1)
      selected_probs = jnp.squeeze(
        # Need ids from pre-mask in last round
          jnp.take_along_axis(probs, jnp.expand_dims(state.final_seqs[:, step-1].astype(jnp.int32), -1), -1), -1)
          # jnp.take_along_axis(probs, jnp.expand_dims(cur_ids.astype(jnp.int32), -1), -1), -1)
    # Ignores the tokens given in the input by overwriting their confidence.
    selected_probs = jnp.where(unknown_map, selected_probs,
                              _CONFIDENCE_OF_KNOWN_TOKENS)

    # Adds noise for randomness
    rng, choice_rng = jax.random.split(rng)
    masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                  choice_temperature * (1. - ratio))
    # Based on previous round's confidence, provide guidance.
    guided_ids = jnp.where(masking, guidance, cur_ids)
    # sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)

    # Calls model on current seqs to get next-iteration seqs.
    logits = tokens_to_logits(guided_ids)
    # Samples the ids using categorical sampling: [batch_size, seq_length].
    sampled_ids = jax.random.categorical(sample_rng, logits)
    sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
    # Updates final seqs with the current sampled_ids.
    final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
    # Use the same pre-determined mask for the next round
    sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)

    return StateWithLogits(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        rng=rng,
        final_seqs=final_seqs,
        logits=logits
        )

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  return final_state.final_seqs

def decode_nondiscard_flax(inputs,
           rng,
           module,
           codebook_size=8192,
           mask_token_id=-1,
           num_iter=12,
           start_iter=0,
           choice_temperature=1.0,
           mask_scheduling_method="cosine"):
  """Fast decoding for iterative generation.

  Args:
    inputs: int32 array: [batch_size, seq_length] input sequence of masked
      tokens, where the masking tokens is defined by mask_token_id.
    rng: jnp.DeviceArray: sampling random state.
    tokens_to_logits: decoder function taking single token slices and cache and
      returning logits and updated cache.
    mask_token_id: int: [Mask] token id.
    num_iter: int: default is 12.
    start_iter: int: default is 0.
    choice_temperature: float: temperature to control the randomness of masking.
    mask_scheduling_method: masking method string. See mask_schedule.py for
      details.

  Returns:
     [batch_size, num_iter, seq_length] output sequence of tokens in all
       iterations.
  """
  inputs = inputs.astype("int32")
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  # Initializes state
  init_state = state_init_with_logits(inputs, rng, num_iter, codebook_size, start_iter=start_iter)

  def loop_cond_fn(mdl, state):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < num_iter)
    return not_at_end

  def loop_body_fn(mdl, state):
    """Beam search loop state update function."""
    rng = state.rng
    step = state.cur_index
    # Current input ids: [batch_size, seq_length].
    cur_ids = state.cur_seqs

    # Calls model on current seqs to get next-iteration seqs.
    logits = mdl(cur_ids)
    logits = logits[..., :codebook_size]
    rng, sample_rng = jax.random.split(rng, 2)
    # Samples the ids using categorical sampling: [batch_size, seq_length].
    sampled_ids = jax.random.categorical(sample_rng, logits)

    # Just updates the masked tokens.
    unknown_map = (cur_ids == mask_token_id)
    sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
    # Defines the mask ratio for the next round. The number to mask out is
    # determined by mask_ratio * unknown_number_in_the_beginning.
    ratio = 1. * (step + 1) / num_iter
    mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                        mask_scheduling_method)
    # Updates final seqs with the current sampled_ids.
    final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
    # Computes the probabilities of each selected tokens.
    probs = jax.nn.softmax(logits, axis=-1)
    selected_probs = jnp.squeeze(
        jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids.astype(jnp.int32), -1), -1), -1)
    # Ignores the tokens given in the input by overwriting their confidence.
    selected_probs = jnp.where(unknown_map, selected_probs,
                               _CONFIDENCE_OF_KNOWN_TOKENS)
    # Gets mask lens for each sample in the batch according to the mask ratio.
    mask_len = jnp.expand_dims(
        jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
    # Keeps at least one of prediction in this round and also masks out at least
    # one and for the next iteration
    mask_len = jnp.maximum(
        1,
        jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

    # Adds noise for randomness
    rng, choice_rng = jax.random.split(rng)
    masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                  choice_temperature * (1. - ratio))
    # Masks tokens with lower confidence.
    sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)
    return StateWithLogits(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        rng=rng,
        final_seqs=final_seqs,
        logits=logits,
    )

  # Run while loop and get final beam search state.

  final_state = flax.linen.while_loop(
    loop_cond_fn, loop_body_fn, module, init_state,
    # carry_variables=[] # don't think I need to carry anything from the module, params are fixed during forward pass.
    # Or maybe I _do_ need to carry them so gradients flow?
  )

  return final_state.logits

def decode_logit_flax_scan(inputs,
           rng,
           module,
           codebook_size=8192,
           mask_token_id=-1,
           num_iter=2,
           start_iter=0,
           choice_temperature=1.0,
           mask_scheduling_method="cosine"):
  """Fast decoding for TUNING iterative generation.

  Args:
    inputs: int32 array: [batch_size, seq_length] input sequence of masked
      tokens, where the masking tokens is defined by mask_token_id.
    rng: jnp.DeviceArray: sampling random state.
    tokens_to_logits: decoder function taking single token slices and cache and
      returning logits and updated cache.
    mask_token_id: int: [Mask] token id.
    num_iter: int: default is 12.
    start_iter: int: default is 0.
    choice_temperature: float: temperature to control the randomness of masking.
    mask_scheduling_method: masking method string. See mask_schedule.py for
      details.

  Returns:
     [batch_size, num_iter, seq_length] output sequence of tokens in all
       iterations.
  """
  inputs = inputs.astype("int32")
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  init_state = state_init_with_logits(inputs, rng, num_iter, codebook_size, start_iter=start_iter)

  def loop_body_fn(mdl, state: StateWithLogits):
      rng = state.rng
      step = state.cur_index
      cur_ids = state.cur_seqs
      # if step == state.final_seqs.shape - 1: # Nop, if statements are illegal.
      #   cur_ids = jax.lax.stop_gradient(cur_ids)

      logits = mdl(cur_ids)
      logits = logits[..., :codebook_size]
      rng, sample_rng = jax.random.split(rng, 2)
      sampled_ids = jax.random.categorical(sample_rng, logits)

      unknown_map = (cur_ids == mask_token_id)
      sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
      ratio = 1. * (step + 1) / num_iter
      mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                          mask_scheduling_method)
      final_seqs = jax.lax.dynamic_update_slice(
          state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
      probs = jax.nn.softmax(logits, axis=-1)
      selected_probs = jnp.squeeze(
          jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids.astype(jnp.int32), -1), -1), -1)
      selected_probs = jnp.where(unknown_map, selected_probs,
                                _CONFIDENCE_OF_KNOWN_TOKENS)
      mask_len = jnp.expand_dims(
          jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
      mask_len = jnp.maximum(
          1,
          jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

      rng, choice_rng = jax.random.split(rng)
      masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                    choice_temperature * (1. - ratio))
      sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)
      return StateWithLogits(
          cur_index=state.cur_index + 1,
          cur_seqs=sampled_ids,
          rng=rng,
          final_seqs=final_seqs,
          logits=logits,
      ), None # don't need anything but final logit which I keep in state

  # Replace the while loop with a call to lax.scan.
  scan = flax.linen.scan(
    loop_body_fn, variable_broadcast="params",
    split_rngs={"params": False},
    length=num_iter - start_iter)
  final_state, _ = scan(module, init_state) # No input, just the carry
  return final_state.logits

def decode_logit_flax_manual(inputs,
           rng,
           module,
           codebook_size=8192,
           mask_token_id=-1,
           num_iter=2,
           start_iter=0,
           choice_temperature=1.0,
           mask_scheduling_method="cosine",
           iterate_guidance=False,
           reweight_guidance=False,
           reweight_kernel=None,
  ):
  """Fast decoding for TUNING iterative generation. Manual unroll. Only supports 2 iterations.
  Stops gradient before all but last iteration

  Args:
    inputs: int32 array: [batch_size, seq_length] input sequence of masked
      tokens, where the masking tokens is defined by mask_token_id.
    rng: jnp.DeviceArray: sampling random state.
    tokens_to_logits: decoder function taking single token slices and cache and
      returning logits and updated cache.
    mask_token_id: int: [Mask] token id.
    num_iter: int: default is 12.
    start_iter: int: default is 0.
    choice_temperature: float: temperature to control the randomness of masking.
    mask_scheduling_method: masking method string. See mask_schedule.py for
      details.

  Returns:
     [batch_size, num_iter, seq_length] output sequence of tokens in all
       iterations.
  """
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  init_state = state_init_with_logits(inputs, rng, num_iter, codebook_size, start_iter=start_iter)


  def loop_body_fn(mdl, state: StateWithLogits, guidance=None, confidence=None, stop_grad=False):
      rng = state.rng
      step = state.cur_index
      cur_ids = state.cur_seqs
      if stop_grad:
        cur_ids = jax.lax.stop_gradient(cur_ids)
      unknown_map = (cur_ids == mask_token_id)
      logits = mdl(cur_ids, guidance_ids=guidance if iterate_guidance else None)
      logits = logits[..., :codebook_size]
      if reweight_guidance and guidance is not None:
        reweight = guidance_to_reweight(reweight_kernel, guidance, confidence)
        reweight = jnp.where(unknown_map[..., None], reweight, 1.)
        logits = logits * reweight
      rng, sample_rng = jax.random.split(rng, 2)
      sampled_ids = jax.random.categorical(sample_rng, logits)

      sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
      ratio = 1. * (step + 1) / num_iter
      mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                          mask_scheduling_method)
      final_seqs = jax.lax.dynamic_update_slice(
          state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
      probs = jax.nn.softmax(logits, axis=-1)
      selected_probs = jnp.squeeze(
          jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids.astype(jnp.int16), -1), -1), -1)
      selected_probs = jnp.where(unknown_map, selected_probs,
                                _CONFIDENCE_OF_KNOWN_TOKENS)
      mask_len = jnp.expand_dims(
          jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
      mask_len = jnp.maximum(
          1,
          jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

      rng, choice_rng = jax.random.split(rng)
      masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                    choice_temperature * (1. - ratio))
      retained_guidance = jnp.where(masking, sampled_ids, mask_token_id)
      sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)
      return StateWithLogits(
          cur_index=state.cur_index + 1,
          cur_seqs=sampled_ids,
          rng=rng,
          final_seqs=final_seqs,
          logits=logits,
      ), jax.lax.stop_gradient(retained_guidance), selected_probs # don't need anything but final logit which I keep in state

  # Replace the while loop with a call to lax.scan.
  scan_state, guidance, selected_probs = loop_body_fn(module, init_state)
  scan_state = loop_body_fn(module, scan_state, guidance=guidance, confidence=selected_probs, stop_grad=True)[0]
  return scan_state.logits