
from dataclasses import dataclass

from sklearn.preprocessing import KBinsDiscretizer, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer, QuantileTransformer, RobustScaler
from sklearn.preprocessing import StandardScaler

from .step import Step


@dataclass
class Scale(Step):
    """Contains numerical scaler ingredients and algorithms."""

    technique : str = 'none'
    techniques : object = None
    parameters : object = None
    runtime_parameters : object = None
    data_to_use : str = 'train'
    name : str = 'scaler'

    def __post_init__(self):
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

    def implement(self, ingredients, columns = None):
        if self.technique != 'none':
            if not columns:
                columns = ingredients.scalers
            self._initialize(select_parameters = True)
            self._store_feature_names(x = ingredients.x)
            if columns:
                ingredients.x[columns] = self.fit_transform(
                        ingredients.x[columns],ingredients.y)
            else:
                ingredients.x = self.fit_transform(
                        ingredients.x, ingredients.y)
            ingredients.x = self._get_feature_names(x = ingredients.x)
        return ingredients