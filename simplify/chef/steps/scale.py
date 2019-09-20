
from dataclasses import dataclass

from sklearn.preprocessing import (KBinsDiscretizer, MaxAbsScaler,
                                   MinMaxScaler, Normalizer, PowerTransformer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)

from simplify.core.base import SimpleStep


@dataclass
class Scale(SimpleStep):
    """Scales numerical data according to selected algorithm.
    
    Args:
        technique(str): name of technique - it should always be 'gauss'
        parameters(dict): dictionary of parameters to pass to selected technique
            algorithm.
        auto_finalize(bool): whether 'finalize' method should be called when the
            class is instanced. This should generally be set to True.
        store_names(bool): whether this class requires the feature names to be
            stored before the 'finalize' and 'produce' methods or called and
            then restored after both are utilized. This should be set to True
            when the class is using numpy methods.
        name(str): name of class for matching settings in the Idea instance and
            for labeling the columns in files exported by Critic.
    """
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    store_names : bool = True
    name : str = 'scaler'

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Public Methods """

    def draft(self):
        self.options = {'bins' : KBinsDiscretizer,
                        'gauss' : Gaussify,
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
        elif self.technique in ['gauss']:
            self.default_parameters = {'standardize' : False,
                                       'copy' : self.parameters['copy']}
        else:
            self.default_parameters = {'copy' : False}
        self.selected_parameters = True
        return self

    def finalize(self):
        self._nestify_parameters()
        self._finalize_parameters()
        if self.technique in ['gauss']:
            self.algorithm = self.options[self.technique](
                parameters = self.parameters)
        elif self.technique != 'none':
            self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def produce(self, ingredients, plan = None, columns = None):
        if self.technique != 'none':
            if not columns:
                columns = ingredients.scalers
            ingredients._store_column_names(x = ingredients.x)
            if self.technique == 'gauss':
                ingredients = self.algorithm.produce(ingredients = ingredients,
                                                   columns = columns)
            else:
                ingredients.x[columns] = self.fit_transform(
                        ingredients.x[columns], ingredients.y)
            ingredients.x = ingredients._get_column_names(x = ingredients.x)
        return ingredients

@dataclass
class Gaussify(SimpleStep):
    """Transforms data columns to more gaussian distribution.
    
    The particular method is chosen between 'box-cox' and 'yeo-johnson' based
    on whether the particular data column has values below zero.
    
    Args:
        technique(str): name of technique - it should always be 'gauss'
        parameters(dict): dictionary of parameters to pass to selected technique
            algorithm.
        custom_algorithm(bool): flag indicating whether this is a class specific
            to the siMpLify package. This flag can be used by various classes
            to properly utilize the class instance methods.
        auto_finalize(bool): whether 'finalize' method should be called when the
            class is instanced. This should generally be set to True.
        name(str): name of class for matching settings in the Idea instance and
            for labeling the columns in files exported by Critic.
    """
    technique : str = 'gauss'
    parameters : object = None
    custom_algorithm : bool = True
    auto_finalize : bool = True
    name : str = 'gaussifier'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {'box-cox' : PowerTransformer,
                        'yeo-johnson' : PowerTransformer}
        return self

    def finalize(self):
        self._nestify_parameters()
        self._finalize_parameters()
        self.positive_tool = PowerTransformer(method = 'box_cox',
                                              **self.parameters)
        self.negative_tool = PowerTransformer(method = 'yeo_johnson',
                                              **self.parameters)
        self.rescaler = MinMaxScaler(copy = self.parameters['copy'])
        return self

    def produce(self, ingredients, columns = None):
        if not columns:
            columns = ingredients.numerics
        for column in columns:
            if ingredients.x[column].min() >= 0:
                ingredients.x[column] = self.positive_tool.fit_transform(
                        ingredients.x[column])
            else:
                ingredients.x[column] = self.negative_tool.fit_transform(
                        ingredients.x[column])
            ingredients.x[column] = self.rescaler.fit_transform(
                    ingredients.x[column])
        return ingredients