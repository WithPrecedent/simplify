

from dataclasses import dataclass

from category_encoders import BackwardDifferenceEncoder, BaseNEncoder
from category_encoders import BinaryEncoder, HashingEncoder, HelmertEncoder
from category_encoders import LeaveOneOutEncoder, OneHotEncoder
from category_encoders import OrdinalEncoder, SumEncoder, TargetEncoder

from .step import Step


@dataclass
class Encode(Step):
    """Contains categorical encoders used in the siMpLify package."""

    technique : str = ''
    parameters : object = None

    def __post_init__(self):
        super().__post_init__()
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

    def blend(self, codex, columns = None):
        if self.technique != 'none':
            if self.verbose:
                print('Encoding categorical data with', self.technique, 'encoder')
            if columns:
                self.runtime_parameters.update({'cols' : columns})
            self.initialize()
            self.algorithm.fit(codex.x, codex.y)
            codex.x_train = self.algorithm.transform(
                    codex.x_train.reset_index(drop = True))
            codex.x_test = self.algorithm.transform(
                    codex.x_test.reset_index(drop = True))
            codex.x = self.algorithm.transform(
                    codex.x.reset_index(drop = True))
        return codex