
from dataclasses import dataclass

from .step import Step

@dataclass
class Custom(Step):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {}
        self.defaults = {}
        self.runtime_params = {}
        return self

    def mix(self, data, runtime_params = None):
        if self.name != 'none':
            if self.verbose:
                print('Applying custom', self.name, 'method')
            if runtime_params:
                self.runtime_params = runtime_params
            self.initialize()
            data.x_train, data.y_train = self.fit_transform(
                    data.x_train, data.y_train)
        return data