

from dataclasses import dataclass

from category_encoders import BackwardDifferenceEncoder, BaseNEncoder
from category_encoders import BinaryEncoder, HashingEncoder, HelmertEncoder
from category_encoders import LeaveOneOutEncoder, OneHotEncoder
from category_encoders import OrdinalEncoder, SumEncoder, TargetEncoder

from .step import Step


@dataclass
class Encode(Step):
    """Contains categorical encoders used in the siMpLify package."""

    technique : str = 'none'
    techniques : object = None
    parameters : object = None
    runtime_parameters : object = None
    data_to_use : str = 'train'
    name : str = 'encoder'

    def __post_init__(self):
        self.techniques = {'backward' : BackwardDifferenceEncoder,
                           'basen' : BaseNEncoder,
                           'binary' : BinaryEncoder,
                           'dummy' : OneHotEncoder,
                           'hashing' : HashingEncoder,
                           'helmert' : HelmertEncoder,
                           'loo' : LeaveOneOutEncoder,
                           'ordinal' : OrdinalEncoder,
                           'sum' : SumEncoder,
                           'target' : TargetEncoder}
        self.defaults = {}
        self.runtime_parameters = {}
        return self

    def implement(self, ingredients, columns = None):
        if self.technique != 'none':
            if not columns:
                columns = ingredients.encoders
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