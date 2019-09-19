
from dataclasses import dataclass

from category_encoders import PolynomialEncoder

from simplify.core.base import SimpleStep


@dataclass
class Mix(SimpleStep):
    """Computes new features using different algorithms selected."""
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name: str = 'mixer'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
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

    def produce(self, ingredients, plan = None, columns = None):
        if self.technique != 'none':
            if not columns:
                columns = ingredients.encoders
            if columns:
                self.runtime_parameters.update({'cols' : columns})
            self.finalize()
            self.algorithm.fit(ingredients.x, ingredients.y)
            self.algorithm.transform(
                    ingredients.x_train).reset_index(drop = True)
            self.algorithm.transform(
                    ingredients.x_test).reset_index(drop = True)
        return ingredients