"""Compatibility alias for registration distance workflows."""
from hausdorff_kernel import *  # noqa: F401,F403
import hausdorff_kernel as _kernel

globals().update({k: v for k, v in vars(_kernel).items() if not k.startswith("__")})
