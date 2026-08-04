from dataclasses import dataclass, fields


@dataclass
class BaseOutput:
    def __post_init__(self):
        pass

    def __getitem__(self, k):
        if isinstance(k, str):
            return getattr(self, k)
        return self.to_tuple()[k]

    def to_tuple(self):
        return tuple(getattr(self, f.name) for f in fields(self))
