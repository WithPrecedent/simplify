
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import mutual_info_regression, RFE, RFECV
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr
from sklearn.feature_selection import SelectFromModel

from .step import Step


@dataclass
class Reduce(Step):
    """Contains feature selectors used in the siMpLify package."""

    technique : str = 'none'
    techniques : object = None
    parameters : object = None
    runtime_parameters : object = None
    data_to_use : str = 'train'
    name : str = 'reducer'

    def __post_init__(self):
        self.options = {'kbest' : SelectKBest,
                        'fdr' : SelectFdr,
                        'fpr' : SelectFpr,
                        'custom' : SelectFromModel,
                        'rfe' : RFE,
                        'rfecv' : RFECV}
        self.defaults = {}
        self.scorers = {'f_classif' : f_classif,
                        'chi2' : chi2,
                        'mutual_class' : mutual_info_classif,
                        'mutual_regress' : mutual_info_regression}
        self.runtime_parameters = {}
        return self

    def _set_param_groups(self, estimator):
#        self.parameters.update({'n_features_to_select' : self.parameters['k']})
        if self.technique == 'rfe':
            self.defaults = {'n_features_to_select' : 10,
                             'step' : 1}
            self.runtime_parameters = {'estimator' : estimator}
        elif self.technique == 'kbest':
            self.defaults = {'k' : 10,
                             'score_func' : f_classif}
            self.runtime_parameters = {}
        elif self.technique in ['fdr', 'fpr']:
            self.defaults = {'alpha' : 0.05,
                             'score_func' : f_classif}
            self.runtime_parameters = {}
        elif self.technique == 'custom':
            self.defaults = {'threshold' : 'mean'}
            self.runtime_parameters = {'estimator' : estimator}
        self.select_parameters(parameters_to_use = self.defaults.keys())
#        if 'score_func' in self.parameters:
#            self.parameters['score_func'] = self.scorers[self.parameters['score_func']]
        return self

    def implement(self, ingredients, estimator = None):
        if self.technique != 'none':
            self._set_param_groups(estimator)
            if self.parameters['score_func']:
                self.parameters['score_func'] = (
                        self.scorers[self.parameters['score_func']])
            self._initialize()
            if 'k' in self.parameters:
                num_features = self.parameters['k']
            else:
                num_features = self.parameters['n_features_to_select']
            if len(ingredients.x_train.columns) > num_features:
                self.algorithm.fit(ingredients.x_train, ingredients.y_train)
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
            else:
                print('Reduce lacks enough columns to reduce columns')
        return ingredients