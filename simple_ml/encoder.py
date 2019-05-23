"""
Encoder is a class containing categorical encoders used in the siMpLify
package.
"""

from dataclasses import dataclass

from category_encoders import BackwardDifferenceEncoder, BaseNEncoder
from category_encoders import BinaryEncoder, HashingEncoder, HelmertEncoder
from category_encoders import LeaveOneOutEncoder, OneHotEncoder
from category_encoders import OrdinalEncoder, SumEncoder, TargetEncoder

from step import Step


@dataclass
class Encoder(Step):

    name : str = ''
    params : object = None

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
        self.runtime_params = {}
        return self

    def mix(self, data, columns = None):
        if self.name != 'none':
            if self.verbose:
                print('Encoding categorical data with', self.name, 'algorithm')
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