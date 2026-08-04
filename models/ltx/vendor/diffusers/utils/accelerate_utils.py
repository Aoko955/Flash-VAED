

from packaging import version

from .import_utils import is_accelerate_available

if is_accelerate_available():
    import accelerate


def apply_forward_hook(method):
    if not is_accelerate_available():
        return method
    accelerate_version = version.parse(accelerate.__version__).base_version
    if version.parse(accelerate_version) < version.parse("0.17.0"):
        return method

    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_hf_hook") and hasattr(self._hf_hook, "pre_forward"):
            self._hf_hook.pre_forward(self)
        return method(self, *args, **kwargs)

    return wrapper
