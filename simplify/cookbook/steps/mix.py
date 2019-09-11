
from dataclasses import dataclass

from category_encoders import PolynomialEncoder

from simplify.core.step import Step


@dataclass
class Mix(Step):
    """Computes new features using different algorithms selected."""
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name: str = 'mixer'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _define(self):
        self.options = {'polynomial' : PolynomialEncoder,
                        'quotient' : self.quotient_features,
                        'sum' : self.sum_features,
                        'difference' : self.difference_features}
        self.default_parameters = {}
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

    def start(self, ingredients, recipe, columns = None):
        if self.technique != 'none':
            if not columns:
                columns = ingredients.encoders
            if columns:
                self.runtime_parameters.update({'cols' : columns})
            self.prepare()
            self.algorithm.fit(ingredients.x, ingredients.y)
            self.algorithm.transform(
                    ingredients.x_train).reset_index(drop = True)
            self.algorithm.transform(
                    ingredients.x_test).reset_index(drop = True)
        return ingredients