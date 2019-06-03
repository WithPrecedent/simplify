from dataclasses import dataclass

from category_encoders import PolynomialEncoder

from .step import Step

@dataclass
class Interactor(Step):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'polynomial' : PolynomialEncoder,
                        'quotient' : self.quotient_features,
                        'sum' : self.sum_features,
                        'difference' : self.difference_features}
        self.defaults = {}
        self.runtime_params = {}
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

    def mix(self, data, columns = None):
        if self.name != 'none':
            if self.verbose:
                print('Creating variables with', self.name, 'interactions')
            if columns:
                self.runtime_params.update({'cols' : columns})
            self.initialize()
            self.algorithm.fit(data.x, data.y)
            data.x_train = self.algorithm.transform(
                    data.x_train.reset_index(drop = True))
            data.x_test = self.algorithm.transform(
                    data.x_test.reset_index(drop = True))
            data.x = self.algorithm.transform(
                    data.x.reset_index(drop = True))
        return data