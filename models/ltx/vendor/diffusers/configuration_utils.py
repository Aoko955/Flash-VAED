
import functools
import inspect
from collections import OrderedDict
from typing import Any, Dict


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            setattr(self, key, value)
        object.__setattr__(self, "__frozen", True)

    def __delitem__(self, *args, **kwargs):
        raise Exception("You cannot use ``__delitem__`` on a FrozenDict")

    def __setitem__(self, key, value):
        if hasattr(self, "__frozen") and object.__getattribute__(self, "__frozen"):
            raise Exception("You cannot use ``__setitem__`` on a FrozenDict")
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):

        object.__setattr__(self, "__frozen", False)
        super().update(*args, **kwargs)
        for key, value in self.items():
            setattr(self, key, value)
        object.__setattr__(self, "__frozen", True)


def register_to_config(init):
    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        sig = inspect.signature(init)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        init_dict = {p.name: p.default for p in params if p.default is not inspect._empty}
        for arg, param in zip(args, params):
            init_dict[param.name] = arg
        init_dict.update(kwargs)
        new_kwargs = {k: v for k, v in init_dict.items() if not str(k).startswith("_")}
        self.register_to_config(**new_kwargs)
        init(self, *args, **kwargs)

    return inner_init


class ConfigMixin:
    config_name = "config.json"

    def register_to_config(self, **kwargs):
        if not hasattr(self, "_internal_dict") or self._internal_dict is None:
            self._internal_dict = FrozenDict(kwargs)
        else:
            d = dict(self._internal_dict)
            d.update(kwargs)
            self._internal_dict = FrozenDict(d)

    @property
    def config(self) -> Dict[str, Any]:
        return getattr(self, "_internal_dict", FrozenDict())
