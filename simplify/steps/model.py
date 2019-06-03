"""
Model is a class containing machine learning algorithms used in the siMpLify
package.
"""

from dataclasses import dataclass

from scipy.stats import randint, uniform
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.cluster import Birch, DBSCAN, KMeans, MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Lasso, LassoLars
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import OneClassSVM, SVC, SVR
#from skopt import BayesSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#from pystan import StanModel
from xgboost import XGBClassifier, XGBRegressor

from .step import Step


@dataclass
class Model(Step):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.settings.localize(instance = self, sections = ['recipes'])
        self._set_options()
        self.defaults = {}
        self._parse_params()
        self.initialize()
        if self.hyperparameter_search:
            self._setup_search()
        return self

    def _set_options(self):
        if self.model_type in ['classifier']:
            self.options = {'adaboost' : AdaBoostClassifier,
                            'baseline_classifier' : DummyClassifier,
                            'logit' : LogisticRegression,
                            'random_forest' : RandomForestClassifier,
#                            'stan' : StanModel,
                            'svm_linear' : SVC,
                            'svm_poly' : SVC,
                            'svm_rbf' : SVC,
                            'svm_sigmoid' : SVC,
#                            'tensor_flow' : KerasClassifier,
#                            'torch' : NeuralNetClassifier,
                            'xgb' : XGBClassifier}
        elif self.model_type in ['regressor']:
            self.options = {'adaboost' : AdaBoostRegressor,
                            'baseline_regressor' : DummyRegressor,
                            'bayes_ridge' : BayesianRidge,
                            'lasso' : Lasso,
                            'lasso_lars' : LassoLars,
                            'ols' : LinearRegression,
                            'random_forest' : RandomForestRegressor,
                            'ridge' : Ridge,
#                            'stan' : StanModel,
                            'svm_linear' : SVR,
                            'svm_poly' : SVR,
                            'svm_rbf' : SVR,
                            'svm_sigmoid' : SVR,
#                            'tensor_flow' : KerasRegressor,
#                            'torch' : NeuralNetRegressor,
                            'xgb' : XGBRegressor}
        elif self.model_type in ['clusterer']:
            self.options = {'affinity' : AffinityPropagation,
                            'agglomerative' : AgglomerativeClustering,
                            'birch' : Birch,
                            'dbscan' : DBSCAN,
                            'kmeans' : KMeans,
                            'mean_shift' : MeanShift,
                            'spectral' : SpectralClustering,
                            'svm_linear' : OneClassSVM,
                            'svm_poly' : OneClassSVM,
                            'svm_rbf' : OneClassSVM,
                            'svm_sigmoid' : OneClassSVM}
        return self

    def _parse_params(self):
        self.hyperparameter_search = False
        self.grid = {}
        new_params = {}
        for param, values in self.params.items():
            if isinstance(values, list):
                self.hyperparameter_search = True
                if self._list_type(values, float):
                    self.grid.update({param : uniform(values[0], values[1])})
                elif self._list_type(values, int):
                    self.grid.update({param : randint(values[0], values[1])})
            else:
                new_params.update({param : values})
        self.params = new_params
        self.runtime_params = {'random_state' : self.seed}
        if 'svm' in self.name:
            self._svm_params()
        elif 'baseline' in self.name:
            self._baseline_params()
        elif 'xgb' in self.name:
            if not hasattr(self, 'scale_pos_weight'):
                self.scale_pos_weight = 1
            if self.gpu:
                self.runtime_params.update({'tree_Step' : 'gpu_exact'})
            if self.hyperparameter_search:
                self.grid.update({'scale_pos_weight' :
                                  uniform(self.scale_pos_weight / 2,
                                  self.scale_pos_weight * 2)})
            else:
                self.params.update(
                        {'scale_pos_weight' : self.scale_pos_weight})
        return self

    def _svm_params(self):
        svm_params = {'svm_linear' : 'linear',
                      'svm_poly' : 'poly',
                      'svm_rbf' : 'rbf',
                      'svm_sigmoid' : 'sigmoid'}
        self.params.update({'kernel' : svm_params[self.name],
                            'probability' : True})
        return self

    def _baseline_params(self):
        if self.name in ['baseline_classifier']:
            self.params.update({'strategy' : 'most_frequent'})
        elif self.name in ['baseline_regressor']:
            self.params.update({'strategy' : 'mean'})
        return self

    def _setup_search(self):
        self.search_options = {'random' : RandomizedSearchCV,
                               'fixed' : GridSearchCV}
#                               'bayes' : BayesSearchCV}
        self.search_runtime_params = {'estimator' : self.algorithm,
                                      'param_distributions' : self.grid,
                                      'random_state' : self.seed}
        self.search_params = self.settings['models_params']
        if self.search_params['refit']:
            self.search_params['scoring'] = (
                    self._listify(self.search_params['scoring'])[0])
        self.search_params.update(self.search_runtime_params)
        self.search_method = self.search_options[self.search_algorithm](
                **self.search_params)
        return self

    def search(self, data):
        if self.verbose:
            print('Searching for best hyperparameters for the',
                  self.name, 'model using', self.search_algorithm,
                  'search algorithm')
        self.search_method.fit(data.x_train, data.y_train)
        self.best_estimator = self.search_method.best_estimator_
        if self.verbose:
            print('The', self.search_params['scoring'],
                  'score of the best estimator for the', self.name,
                  'model is', f'{self.search_method.best_score_ : 4.4f}')
        return self

    def mix(self, data):
        if self.name != 'none':
            if self.verbose:
                print('Applying', self.name, 'model to data')
            if self.hyperparameter_search:
                self.search(data)
                self.algorithm = self.best_estimator
            else:
                self.algorithm.fit(data.x_train, data.y_train)
        return self

    def fit(self, x, y):
        if self.hyperparameter_search:
            self.search(x, y)
            self.algorithm = self.best_estimator
        else:
            self.algorithm.fit(x, y)
        return self

    def transform(self, x, y):
        error = 'transform is not implemented for machine learning models'
        raise NotImplementedError(error)
        return self

    def fit_transform(self, x, y):
        error = 'fit_transform is not implemented for machine learning models'
        raise NotImplementedError(error)
        return self