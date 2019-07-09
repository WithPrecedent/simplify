
from dataclasses import dataclass

from sklearn.preprocessing import (KBinsDiscretizer, MaxAbsScaler,
                                   MinMaxScaler, Normalizer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)

from .cookbook_step import CookbookStep


@dataclass
class Scale(CookbookStep):
    """Scales numerical data according to selected algorithm."""
    technique : str = ''
    techniques : object = None
    parameters : object = None
    runtime_parameters : object = None
    auto_prepare : bool = True
    name : str = 'scaler'

    def __post_init__(self):
        self._set_defaults()
        super().__post_init__()
        return self

    def _set_defaults(self):
        if not self.techniques:
            self.techniques = {'bins' : KBinsDiscretizer,
                               'maxabs' : MaxAbsScaler,
                               'minmax' : MinMaxScaler,
                               'normalizer' : Normalizer,
                               'quantile' : QuantileTransformer,
                               'robust' : RobustScaler,
                               'standard' : StandardScaler}
        if self.technique in ['bins']:
            self.default_parameters = {'copy' : False,
                                       'encode' : 'ordinal',
                                       'strategy' : 'uniform',
                                       'threshold' : 1}
        else:
            self.default_parameters = {'copy' : False}
        self.selected_parameters = True
        return self

    def start(self, ingredients, recipe = None, columns = None):
        if self.technique != 'none':
            if not columns:
                columns = ingredients.scalers
            self._store_feature_names(x = ingredients.x)
            if columns:
                ingredients.x[columns] = self.fit_transform(
                        ingredients.x[columns],ingredients.y)
            else:
                ingredients.x = self.fit_transform(
                        ingredients.x, ingredients.y)
            ingredients.x = self._get_feature_names(x = ingredients.x)
        return ingredients