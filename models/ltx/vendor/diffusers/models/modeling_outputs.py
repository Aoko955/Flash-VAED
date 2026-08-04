from dataclasses import dataclass


@dataclass
class BaseOutput:
    def __getitem__(self, k):
        if isinstance(k, str):
            return getattr(self, k)
        return self.to_tuple()[k]

    def to_tuple(self):
        return tuple(getattr(self, f.name) for f in self.__dataclass_fields__.values())


@dataclass
class AutoencoderKLOutput(BaseOutput):
    latent_dist: object
