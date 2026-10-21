import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    x = np.asarray(scores, dtype=float)
    n = x.shape[-1]

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    return np.where(mask, mask_value, x)