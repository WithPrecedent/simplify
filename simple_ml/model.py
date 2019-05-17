"""
Model is a class containing machine learning algorithms used in the siMpLify
package.
"""


from dataclasses import dataclass
from scipy.stats import randint, uniform
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.cluster import Birch, DBSCAN, KMeans, MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Lasso, LassoLars
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import OneClassSVM, SVC, SVR

#from skopt import BayesSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#from pystan import StanModel
#from skorch import NeuralNetClassifier, NeuralNetRegressor
from xgboost import XGBClassifier, XGBRegressor

from step import Step

@dataclass
class Model(Step):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        if self.model_type in ['classifier']:
            self.options = {'ada' : AdaBoostClassifier,
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
            self.options = {'ada' : AdaBoostRegressor,
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
        elif self.model_type in ['unsupervised']:
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
        self._parse_params()
        self.initialize()
        if self.hyperparameter_search:
            self._setup_search()
        return self

    def _svm_params(self):
        svm_params = {'svm_linear' : 'linear',
                      'svm_poly' : 'poly',
                      'svm_rbf' : 'rbf',
                      'svm_sigmoid' : 'sigmoid'}
        self.params.update({'kernel' : svm_params[self.name]})
        return self

    def _setup_search(self):
        self.search_options = {'random' : RandomizedSearchCV,
                               'fixed' : GridSearchCV}
#                               'bayes' : BayesSearchCV}
        self.search_runtime_params = {'estimator' : self.algorithm,
                                      'param_distributions' : self.grid,
                                      'random_state' : self.seed}
        self.search_params = self.settings['search_params']
        if self.search_params['refit']:
            self.search_params['scoring'] = self._listify(
                    self.search_params['scoring'])[0]
        self.search_params.update(self.search_runtime_params)
        self.search_method = self.search_options[self.search_algorithm](
                **self.search_params)
        return self

    def _parse_params(self):
        self.use_grid = False
        self.grid = {}
        new_params = {}
        for param, values in self.params.items():
            if isinstance(values, list):
                self.use_grid = True
                if self._list_type(values, float):
                    self.grid.update({param : uniform(values[0], values[1])})
                elif self._list_type(values, int):
                    self.grid.update({param : randint(values[0], values[1])})
            else:
                new_params.update({param : values})
        self.params = new_params
        if 'svm' in self.name:
            self._svm_params()
        self.runtime_params = {'random_state' : self.seed}
        if self.name == 'xgb':
            if not hasattr(self, 'scale_pos_weight'):
                self.scale_pos_weight = 1
            if self.gpu:
                self.runtime_params.update({'tree_Step' : 'gpu_exact'})
            if self.use_grid:
                self.grid.update({'scale_pos_weight' :
                                  uniform(self.scale_pos_weight / 2,
                                  self.scale_pos_weight * 2)})
            else:
                self.params.update(
                        {'scale_pos_weight' : self.scale_pos_weight})
        return self

    def search(self, x, y):
        if self.verbose:
            print('Searching for best hyperparameters for the',
                  self.name, 'model using', self.search_algorithm,
                  'search algorithm')
        self.search_method.fit(x, y)
        self.best = self.search_method.best_estimator_
        print('The', self.search_params['scoring'],
              'score of the best estimator for the', self.name,
              'model is', str(self.search_method.best_score_))
        return self