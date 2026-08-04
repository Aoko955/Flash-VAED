__version__ = "0.35.0.dev0-flashvaed"

from .deprecation_utils import deprecate
from .import_utils import is_accelerate_available, is_torch_npu_available, is_torch_version
from .logging import get_logger
from .outputs import BaseOutput

__all__ = [
    "BaseOutput",
    "deprecate",
    "get_logger",
    "is_accelerate_available",
    "is_torch_npu_available",
    "is_torch_version",
]
