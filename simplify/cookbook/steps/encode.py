
from dataclasses import dataclass

from category_encoders import (BackwardDifferenceEncoder, BaseNEncoder,
                               BinaryEncoder, HashingEncoder, HelmertEncoder,
                               LeaveOneOutEncoder, OneHotEncoder,
                               OrdinalEncoder, SumEncoder, TargetEncoder)

from simplify.core.step import Step


@dataclass
class Encode(Step):
    """Encodes categorical variables according to selected algorithms."""
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'encoder'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _define(self):
        self.options = {'backward' : BackwardDifferenceEncoder,
                        'basen' : BaseNEncoder,
                        'binary' : BinaryEncoder,
                        'dummy' : OneHotEncoder,
                        'hashing' : HashingEncoder,
                        'helmert' : HelmertEncoder,
                        'loo' : LeaveOneOutEncoder,
                        'ordinal' : OrdinalEncoder,
                        'sum' : SumEncoder,
                        'target' : TargetEncoder}
        self.default_parameters = {}
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