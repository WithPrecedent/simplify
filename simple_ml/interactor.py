from dataclasses import dataclass

from category_encoders import PolynomialEncoder

from step import Step


@dataclass
class Interactor(Step):

    name : str = ''
    params : object = None
    columns : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'polynomial' : PolynomialEncoder,
                        'quotient' : self.quotient_features,
                        'sum' : self.sum_features,
                        'difference' : self.difference_features}
        self.defaults = {}
        self.runtime_params = {'cols' : self.columns}
        self.initialize()
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