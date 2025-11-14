import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_global_seed(seed: int = 42):
    """
    Set all major random seeds to ensure deterministic results across
    Python, NumPy, and PyTorch (CPU/GPU).
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Python hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Global seed fixed at {seed}")
