"""Compatibility name for callers describing point-set distance operators."""
from hausdorff_kernel import *  # noqa: F401,F403
import hausdorff_kernel as _kernel

globals().update({k: v for k, v in vars(_kernel).items() if not k.startswith("__")})
