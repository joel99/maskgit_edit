r"""
    A Flax tuning script that mimics the MaskGIT MVTM training process, with the following changes:
    1. N-step decoding DURING training (vs 1-step)
    2. Add input encoding for all tokens, including masked ones.
        - TODO experiment whether the input content for masked tokens should be directly added or provided separately.
    3. Scale mask embedding according to confidence.
"""

# We keep identically the recommended training settings from the paper, but have to reproduce the training loop since they didn't release it yet.


r"""
    Tasklog
    # 0. dataloader
    # loads images, masks, and strokes. For starters we can play with just the imgs and start from step 0.
    # 1. VQ-encoding to create inputs and outputs
    #   - output targets should be on source image
    #   - we initialize at step 2, just like inference, and lock in fully unmasked
    # 2. Identify label smoothing and/or their hps
    # 3. ? train?
"""