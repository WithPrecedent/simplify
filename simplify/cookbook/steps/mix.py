
from dataclasses import dataclass

from category_encoders import PolynomialEncoder

from .step import Step

@dataclass
class Mix(Step):
    """Contains algorithms for testing variable interactions in the siMpLify
    package.
    """

    technique : str = ''
    parameters : object = None

    def __post_init__(self):
        super().__post_init__()
        self.techniques = {'polynomial' : PolynomialEncoder,
                        'quotient' : self.quotient_features,
                        'sum' : self.sum_features,
                        'difference' : self.difference_features}
        self.defaults = {}
        self.runtime_parameters = {}
        return self

    def quotient_features(self):
        pass
        return self

    def sum_features(self):
        pass
        return self

    def difference_features(self):
        pass
        return self

    def blend(self, codex, columns = None):
        if self.technique != 'none':
            if self.verbose:
                print('Creating variables with', self.technique,
                      'interactions')
            if columns:
                self.runtime_parameters.update({'cols' : columns})
            self.initialize()
            self.algorithm.fit(codex.x, codex.y)
            codex.x_train = self.algorithm.transform(
                    codex.x_train.reset_index(drop = True))
            codex.x_test = self.algorithm.transform(
                    codex.x_test.reset_index(drop = True))
            codex.x = self.algorithm.transform(
                    codex.x.reset_index(drop = True))
        return codex