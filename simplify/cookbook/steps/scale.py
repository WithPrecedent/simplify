
from dataclasses import dataclass

from sklearn.preprocessing import KBinsDiscretizer, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer, QuantileTransformer, RobustScaler
from sklearn.preprocessing import StandardScaler

from .step import Step


@dataclass
class Scale(Step):
    """Contains numerical scaler ingredients and algorithms."""

    technique : str = 'none'
    parameters : object = None

    def __post_init__(self):
        super().__post_init__()
        self.techniques = {'bins' : KBinsDiscretizer,
                           'maxabs' : MaxAbsScaler,
                           'minmax' : MinMaxScaler,
                           'normalizer' : Normalizer,
                           'quantile' : QuantileTransformer,
                           'robust' : RobustScaler,
                           'standard' : StandardScaler}
        if self.technique in ['bins']:
            self.defaults = {'copy' : False,
                             'encode' : 'ordinal',
                             'strategy' : 'uniform',
                             'threshold' : 1}
        else:
            self.defaults = {'copy' : False}
        self.runtime_parameters = {}
        return self

    def blend(self, codex, columns = None):
        if self.technique != 'none':
            if self.verbose:
                print('Scaling numerical columns with', self.technique,
                      'method')
            self.initialize(select_parameters = True)
            if columns:
                codex.x[columns] = self.fit_transform(codex.x[columns],
                                                      codex.y)
            else:
                codex.x = self.fit_transform(codex.x, codex.y)
        return codex