
from dataclasses import dataclass

from sklearn.preprocessing import (KBinsDiscretizer, MaxAbsScaler,
                                   MinMaxScaler, Normalizer, PowerTransformer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)

from ..cookbook_step import CookbookStep
from ...core.technique import Technique


@dataclass
class Scale(CookbookStep):
    """Scales numerical data according to selected algorithm."""
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'scaler'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _define(self):
        self.options = {'bins' : KBinsDiscretizer,
#                        'gauss' : Gaussify,
                        'maxabs' : MaxAbsScaler,
                        'minmax' : MinMaxScaler,
                        'normalizer' : Normalizer,
                        'quantile' : QuantileTransformer,
                        'robust' : RobustScaler,
                        'standard' : StandardScaler}
        if self.technique in ['bins']:
            self.default_parameters = {'encode' : 'ordinal',
                                       'strategy' : 'uniform',
                                       'n_bins' : 5}
        else:
            self.default_parameters = {'copy' : False}
        self.selected_parameters = True
        return self

    def prepare(self):
        """Adds parameters to algorithm and sets import/export folders."""
#        self.conform()
        self._check_parameters()
        self._select_parameters()
        if self.technique == 'gauss':
            self.algorithm = self.options[self.technique](
                    technique = self.technique,
                    parameters = self.parameters)
        elif self.technique != 'none':
            self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def start(self, ingredients, recipe = None, columns = None):
        if self.technique != 'none':
            if not columns:
                columns = ingredients.scalers
            self._store_feature_names(x = ingredients.x)
            if self.technique == 'gauss':
                ingredients = self.algorithm.start(ingredients = ingredients,
                                                   columns = columns)
            else:
                ingredients.x[columns] = self.fit_transform(
                        ingredients.x[columns], ingredients.y)
#            else:
#                ingredients.x = self.fit_transform(
#                        ingredients.x, ingredients.y)
            ingredients.x = self._get_feature_names(x = ingredients.x)
        return ingredients
#
#@dataclass
#class Gaussify(Technique):
#
#    technique : str = ''
#    parameters : object = None
#    auto_prepare : bool = True
#
#    def __post_init__(self):
#        super().__post_init__()
#        return self
#
#    def _define(self):
#        self.options = {'box-cox' : PowerTransformer,
#                        'yeo-johnson' : PowerTransformer}
#        self.default_parameters = {'standardize' : False,
#                                   'copy' : self.parameters['copy']}
#        return self
#
#    def prepare(self):
#        self.positive_tool = PowerTransformer(method = 'box_cox',
#                                              **self.default_parameters)
#        self.negative_tool = PowerTransformer(method = 'yeo_johnson',
#                                              **self.default_parameters)
#        self.rescaler = MinMaxScaler(copy = self.parameters['copy'])
#        return self
#
#    def start(self, ingredients, columns):
#        for column in columns:
#            if ingredients.x[column].min() >= 0:
#                ingredients.x[column] = self.positive_tool.fit_transform(
#                        ingredients.x[column])
#            else:
#                ingredients.x[column] = self.negative_tool.fit_transform(
#                        ingredients.x[column])
#            ingredients.x[column] = self.rescaler.fit_transform(
#                    ingredients.x[column])
#        return ingredients