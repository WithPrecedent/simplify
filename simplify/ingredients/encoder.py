

from dataclasses import dataclass

from category_encoders import BackwardDifferenceEncoder, BaseNEncoder
from category_encoders import BinaryEncoder, HashingEncoder, HelmertEncoder
from category_encoders import LeaveOneOutEncoder, OneHotEncoder
from category_encoders import OrdinalEncoder, SumEncoder, TargetEncoder

from .ingredient import Ingredient


@dataclass
class Encoder(Ingredient):
    """Contains categorical encoders used in the siMpLify package."""

    technique : str = ''
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

    def mix(self, codex, columns = None):
        if self.technique != 'none':
            if self.verbose:
                print('Encoding categorical data with', self.technique, 'encoder')
            if columns:
                self.runtime_params.update({'cols' : columns})
            self.initialize()
            self.algorithm.fit(codex.x, codex.y)
            codex.x_train = self.algorithm.transform(
                    codex.x_train.reset_index(drop = True))
            codex.x_test = self.algorithm.transform(
                    codex.x_test.reset_index(drop = True))
            codex.x = self.algorithm.transform(
                    codex.x.reset_index(drop = True))
        return codex