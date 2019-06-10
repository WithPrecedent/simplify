
from dataclasses import dataclass

from .ingredient import Ingredient

@dataclass
class Custom(Ingredient):

    technique : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {}
        self.defaults = {}
        self.runtime_params = {}
        return self

    def mix(self, codex, runtime_params = None):
        if self.technique != 'none':
            if self.verbose:
                print('Applying custom', self.technique, 'method')
            if runtime_params:
                self.runtime_params = runtime_params
            self.initialize()
            codex.x_train, codex.y_train = self.fit_transform(
                    codex.x_train, codex.y_train)
        return codex