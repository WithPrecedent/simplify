"""
Scaler is a class containing numerical scalers used in the siMpLify package.
"""

from dataclasses import dataclass
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.preprocessing import StandardScaler

from simplify.step import Step


@dataclass
class Scaler(Step):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'maxabs' : MaxAbsScaler,
                        'minmax' : MinMaxScaler,
                        'normalizer' : Normalizer,
                        'quantile' : QuantileTransformer,
                        'robust' : RobustScaler,
                        'standard' : StandardScaler}
        self.defaults = {'copy' : False}
        self.runtime_params = {}
        self.initialize()
        return self