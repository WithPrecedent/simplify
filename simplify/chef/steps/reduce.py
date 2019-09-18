
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_selection import (chi2, f_classif, mutual_info_classif,
                                       mutual_info_regression, RFE, RFECV,
                                       SelectKBest, SelectFdr, SelectFpr,
                                       SelectFromModel)

from simplify.core.base import SimpleStep

@dataclass
class Reduce(SimpleStep):
    """Reduces features using different algorithms, including the model
    algorithm.
    """
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'reducer'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
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
        self.selected_parameters = True
        return self

    def _set_parameters(self, estimator):
        if self.technique in ['rfe', 'rfecv']:
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
        self._finalize_parameters()
        self._select_parameters()
        self.parameters.update({'estimator' : estimator})
        if 'k' in self.parameters:
            self.num_features = self.parameters['k']
        else:
            self.num_features = self.parameters['n_features_to_select']
        return self

    def finalize(self):
        """All preparation has to be at runtime for reduce because of the
        possible inclusion of a fit estimator."""
        pass
        return self

    def produce(self, ingredients, recipe, estimator = None):
        if self.technique != 'none':
            if not estimator:
                estimator = recipe.model.algorithm
            self._set_parameters(estimator)
            self.algorithm = self.options[self.technique](**self.parameters)
            if len(ingredients.x_train.columns) > self.num_features:
                self.algorithm.fit(ingredients.x_train, ingredients.y_train)
                mask = ~self.algorithm.get_support()
                ingredients.drop_columns(df = ingredients.x_train, mask = mask)
                ingredients.drop_columns(df = ingredients.x_test, mask = mask)
        return ingredients