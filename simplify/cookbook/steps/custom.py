
from dataclasses import dataclass

from .step import Step

@dataclass
class Custom(Step):

    ingredient : str = ''
    technique : object = None
    parameters : object = None
    runtime_parameters : object = None
    return_codex : bool = True

    def __post_init__(self):
        super().__post_init__()
        self._set_custom()
        self.techniques = {}
        self.defaults = {}
        self.runtime_parameters = {}
        return self

    def _set_custom(self):
        if not self.runtime_parameters:
            self.runtime_parameters = {}

    def blend(self, codex, runtime_parameters = None):
        if self.technique != 'none':
            if self.verbose:
                print('Applying custom', self.technique, 'method')
            if runtime_parameters:
                self.runtime_parameters = runtime_parameters
            self.initialize()
            codex.x_train, codex.y_train = self.fit_transform(
                    codex.x_train, codex.y_train)
        return codex