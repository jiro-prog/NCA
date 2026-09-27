"""乱数のシードを固定して、同じ結果を再現できるようにする。"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    # cuBLAS を決定的に動かすための設定。CUDA を初めて使う前に入れておく必要がある
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     # CUDA の乱数もまとめて固定される
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
