"""
Encoder is a class containing categorical encoders used in the siMpLify
package.
"""

from dataclasses import dataclass

from category_encoders import BackwardDifferenceEncoder, BaseNEncoder
from category_encoders import BinaryEncoder, HashingEncoder, HelmertEncoder
from category_encoders import LeaveOneOutEncoder, OneHotEncoder
from category_encoders import OrdinalEncoder, SumEncoder, TargetEncoder

from simplify.step import Step


@dataclass
class Encoder(Step):

    name : str = ''
    params : object = None
    columns : object = None

    def __post_init__(self):
        super().__post_init__()
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
        self.defaults = {}
        self.runtime_params = {'cols' : self.columns}
        self.initialize()
        return self

    def fit(self, x, y):
        return self.algorithm.fit(x, y)

    def transform(self, x):
        x = self.algorithm.transform(x)
        for column in self.columns:
            if column in x.columns:
                x[column] = x[column].astype(float, copy = False)
        return x