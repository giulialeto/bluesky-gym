"""
Reproducible seeding for training and evaluation.

One base seed is expanded into independent, non-overlapping seeds with
`np.random.SeedSequence`. The environments draw their randomness (obstacles,
aircraft, waypoints) from their own generator, `self.np_random`,
which is seeded through `env.reset(seed=...)`.

    derive_seed(42, "model")       -> seed of the policy initialisation / SB3
    derive_seed(42, "env")         -> seed of the training environments
    derive_seed(42, "eval", i)     -> seed of the i-th evaluation episode
"""

import random
import numpy as np

def derive_seed(base_seed: int, *keys) -> int:
    """Derive an independent 32-bit seed from `base_seed` and a path of non-negative int/str keys.

    Strings are hashed to ints deterministically (not with `hash()`, which is randomised per process).
    Different keys or different base seeds give unrelated seeds.
    """
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    entropy = [int(base_seed)]
    for key in keys:
        if isinstance(key, str):
            key = int.from_bytes(key.encode("utf-8"), "little")
        if key < 0:
            raise ValueError("seed keys must be non-negative")
        entropy.append(int(key))
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def seed_everything(seed: int) -> None:
    """Seed the global python, numpy and torch generators (policy initialisation, SB3, anything left global)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
