
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
    params : object = None

    def __post_init__(self):
        super().__post_init__()
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
        self.runtime_params = {}
        return self

    def _set_param_groups(self, estimator):
#        self.params.update({'n_features_to_select' : self.params['k']})
        if self.technique == 'rfe':
            self.defaults = {'n_features_to_select' : 10,
                             'step' : 1}
            self.runtime_params = {'estimator' : estimator}
        elif self.technique == 'kbest':
            self.defaults = {'k' : 10,
                             'score_func' : f_classif}
            self.runtime_params = {}
        elif self.technique in ['fdr', 'fpr']:
            self.defaults = {'alpha' : 0.05,
                             'score_func' : f_classif}
            self.runtime_params = {}
        elif self.technique == 'custom':
            self.defaults = {'threshold' : 'mean'}
            self.runtime_params = {'estimator' : estimator}
        self.select_params(params_to_use = self.defaults.keys())
#        if 'score_func' in self.params:
#            self.params['score_func'] = self.scorers[self.params['score_func']]
        return self

    def mix(self, codex, estimator = None):
        if self.technique != 'none':
            if self.verbose:
                print('Using', self.technique, 'for feature reduction')
            self._set_param_groups(estimator)
            if self.params['score_func']:
                self.params['score_func'] = (
                        self.scorers[self.params['score_func']])
            self.initialize()
            if 'k' in self.params:
                num_features = self.params['k']
            else:
                num_features = self.params['n_features_to_select']
            if len(codex.x_train.columns) > num_features:
                self.algorithm.fit(codex.x_train, codex.y_train)
                mask = self.algorithm.get_support()
                new_features = codex.x_train.columns[mask]
                codex.x_train = self.algorithm.transform(codex.x_train)
                codex.x_train = pd.DataFrame(codex.x_train,
                                            columns = new_features)
                codex.x_test = pd.DataFrame(codex.x_test,
                                           columns = new_features)
                codex.x = pd.DataFrame(codex.x,
                                      columns = new_features)
            else:
                print('Selector lacks enough columns to reduce columns')
        return codex