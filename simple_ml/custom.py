
from dataclasses import dataclass

from step import Step

@dataclass
class Custom(Step):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {}
        self.defaults = {}
        self.runtime_params = {}
        self.initialize()
        return self