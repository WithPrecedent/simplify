"""
Selector is a class containing feature selectors used in the siMpLify package.
"""

from dataclasses import dataclass
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import mutual_info_regression, RFE, RFECV
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr
from sklearn.feature_selection import SelectFromModel

from simplify.step import Step


@dataclass
class Selector(Step):

    name : str = ''
    model : object = None
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
        self._set_param_groups()
        self.initialize()
        return self

    def _set_param_groups(self):
        if self.name == 'rfe':
            self.defaults = {'n_features_to_select' : 30,
                             'step' : 1}
            self.runtime_params = {'estimator' : self.model.algorithm}
        elif self.name == 'kbest':
            self.defaults = {'k' : 30,
                             'score_func' : f_classif}
            self.runtime_params = {}
        elif self.name in ['fdr', 'fpr']:
            self.defaults = {'alpha' : 0.05,
                             'score_func' : f_classif}
            self.runtime_params = {}
        elif self.name == 'custom':
            self.defaults = {'threshold' : 'mean'}
            self.runtime_params = {'estimator' : self.model.algorithm}
        self.select_params(params_to_use = self.defaults.keys())
#        if 'score_func' in self.params:
#            self.params['score_func'] = self.scorers[self.params['score_func']]
        return self

    def transform(self, x):
        if len(x.columns) > self.params['n_features_to_select']:
            return self.Step.transform(x)
        else:
            return x