
from torch import nn


class ModelMixin(nn.Module):


    _supports_gradient_checkpointing = False

    def __init__(self):
        super().__init__()
