
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_selection import (chi2, f_classif, mutual_info_classif,
                                       mutual_info_regression, RFE, RFECV,
                                       SelectKBest, SelectFdr, SelectFpr,
                                       SelectFromModel)

from ..cookbook_step import CookbookStep


@dataclass
class Reduce(CookbookStep):
    """Reduces features using different algorithms, including the model
    algorithm.
    """
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'reducer'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        self.options  = {'kbest' : SelectKBest,
                         'fdr' : SelectFdr,
                         'fpr' : SelectFpr,
                         'custom' : SelectFromModel,
                         'rfe' : RFE,
                         'rfecv' : RFECV}
        self.scorers = {'f_classif' : f_classif,
                        'chi2' : chi2,
                        'mutual_class' : mutual_info_classif,
                        'mutual_regress' : mutual_info_regression}
        return self

    def _add_parameters(self, estimator):
        if self.technique == 'rfe':
            self.default_parameters = {'n_features_to_select' : 10,
                                       'step' : 1}
            self.runtime_parameters = {'estimator' : estimator}
        elif self.technique == 'kbest':
            self.default_parameters = {'k' : 10,
                                       'score_func' : f_classif}
            self.runtime_parameters = {}
        elif self.technique in ['fdr', 'fpr']:
            self.default_parameters = {'alpha' : 0.05,
                                       'score_func' : f_classif}
            self.runtime_parameters = {}
        elif self.technique == 'custom':
            self.default_parameters = {'threshold' : 'mean'}
            self.runtime_parameters = {'estimator' : estimator}
        return self

    def start(self, ingredients, recipe, estimator = None):
        if self.technique != 'none':
            if not estimator:
                estimator = recipe.model.algorithm
            self._add_parameters(estimator)
            self.prepare()
            if self.parameters['score_func']:
                self.parameters['score_func'] = (
                        self.scorers[self.parameters['score_func']])
            if 'k' in self.parameters:
                self.num_features = self.parameters['k']
            else:
                self.num_features = self.parameters['n_features_to_select']
            if recipe.data_to_use in ['full']:
                if len(ingredients.x.columns) > self.num_features:
                    self.algorithm.fit(ingredients.x, ingredients.y)
                    mask = self.algorithm.get_support()
                    new_features = ingredients.x.columns[mask]
                    ingredients.x = self.algorithm.transform(ingredients.x)
                    ingredients.x = pd.DataFrame(ingredients.x,
                                                 columns = new_features)
            else:
                if len(ingredients.x_train.columns) > self.num_features:
                    self.algorithm.fit(ingredients.x_train,
                                       ingredients.y_train)
                    mask = self.algorithm.get_support()
                    new_features = ingredients.x_train.columns[mask]
                    ingredients.x_train = self.algorithm.transform(
                            ingredients.x_train)
                    ingredients.x_train = pd.DataFrame(ingredients.x_train,
                                                       columns = new_features)
                    ingredients.x_test = pd.DataFrame(ingredients.x_test,
                                                      columns = new_features)
                    ingredients.x = pd.DataFrame(ingredients.x,
                                                 columns = new_features)
        return ingredients