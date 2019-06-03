

from dataclasses import dataclass

from sklearn.preprocessing import KBinsDiscretizer, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer, QuantileTransformer, RobustScaler
from sklearn.preprocessing import StandardScaler

from .step import Step


@dataclass
class Scaler(Step):
    """Contains numerical scaler ingredients and algorithms.
    """
    name : str = 'none'
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'bins' : KBinsDiscretizer,
                        'maxabs' : MaxAbsScaler,
                        'minmax' : MinMaxScaler,
                        'normalizer' : Normalizer,
                        'quantile' : QuantileTransformer,
                        'robust' : RobustScaler,
                        'standard' : StandardScaler}
        if self.name in ['bins']:
            self.defaults = {'copy' : False,
                             'encode' : 'ordinal',
                             'strategy' : 'uniform',
                             'threshold' : 1}
        else:
            self.defaults = {'copy' : False}
        self.runtime_params = {}
        return self

    def mix(self, data, columns = None):
        if self.name != 'none':
            if self.verbose:
                print('Scaling numerical columns with', self.name, 'method')
            self.initialize(select_params = True)
            if columns:
                data.x[columns] = self.fit_transform(data.x[columns], data.y)
            else:
                data.x = self.fit_transform(data.x, data.y)
        return data