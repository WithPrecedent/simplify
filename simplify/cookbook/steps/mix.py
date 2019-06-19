
from dataclasses import dataclass

from category_encoders import PolynomialEncoder

from .step import Step

@dataclass
class Mix(Step):
    """Contains algorithms for testing variable interactions in the siMpLify
    package.
    """

    technique : str = 'none'
    name: str = 'mixer'

    def __post_init__(self):
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

    def blend(self, ingredients, columns = None):
        if self.technique != 'none':
            if not columns:
                columns = ingredients.mixers
            if columns:
                self.runtime_parameters.update({'cols' : columns})
            self._initialize()
            self.algorithm.fit(ingredients.x, ingredients.y)
            ingredients.x_train = self.algorithm.transform(
                    ingredients.x_train.reset_index(drop = True))
            ingredients.x_test = self.algorithm.transform(
                    ingredients.x_test.reset_index(drop = True))
            ingredients.x = self.algorithm.transform(
                    ingredients.x.reset_index(drop = True))
        return ingredients