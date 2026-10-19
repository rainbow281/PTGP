"""Principal Task Gradient Projection (PTGP)."""

from .core import PTGP, make_balanced_groups
from .data import load_dataset_splits

__all__ = ["PTGP", "make_balanced_groups", "load_dataset_splits"]
