"""J-lens: Jacobian lens and J-space (global workspace) tools for HuggingFace causal LMs."""
from .interventions import ablate, ablate_jspace, steer, swap
from .jspace import Decomposition, JSpace
from .lens import JacobianLens
from .model import ModelAdapter
from .monitor import InjectionMonitor, InjectionReport

__all__ = [
    "ModelAdapter", "JacobianLens", "JSpace", "Decomposition",
    "steer", "ablate", "swap", "ablate_jspace",
    "InjectionMonitor", "InjectionReport",
]
