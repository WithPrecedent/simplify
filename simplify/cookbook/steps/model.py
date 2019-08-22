
from dataclasses import dataclass

from scipy.stats import randint, uniform
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, KMeans, MeanShift,
                             SpectralClustering)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (BayesianRidge, Lasso, LassoLars,
                                  LinearRegression, LogisticRegression, Ridge)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import OneClassSVM, SVC, SVR
#from skopt import BayesSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#from pystan import StanModel
from xgboost import XGBClassifier, XGBRegressor

from ...implements.tools import listify
from ..cookbook_step import CookbookStep

@dataclass
class Model(CookbookStep):
    """Applies machine learning algorithms based upon user selections."""
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'model'

    def __post_init__(self):
        super().__post_init__()
        if self.hyperparameter_search:
            self._prepare_search()
        return self

    def _baseline_parameters(self):
        if self.technique in ['baseline_classifier']:
            self.parameters.update({'strategy' : 'most_frequent'})
        elif self.technique in ['baseline_regressor']:
            self.parameters.update({'strategy' : 'mean'})
        return self

    def _check_parameters(self):
        """Checks if parameters exists. If not, defaults are used. If there
        are no defaults, an empty dict is created for parameters.
        """
        if hasattr(self, 'menu') and self.technique in self.menu.config:
            self.parameters = self.menu.config[self.technique]
        elif hasattr(self, 'default_parameters'):
            self.parameters = self.default_parameters
        else:
            self.parameters = {}
        return self

    def _parse_parameters(self):
        self.hyperparameter_search = False
        self.grid = {}
        new_parameters = {}
        for param, values in self.parameters.items():
            if isinstance(values, list):
                self.hyperparameter_search = True
                if self._list_type(values, float):
                    self.grid.update({param : uniform(values[0], values[1])})
                elif self._list_type(values, int):
                    self.grid.update({param : randint(values[0], values[1])})
            else:
                new_parameters.update({param : values})
        self.parameters = new_parameters
        self.runtime_parameters = {'random_state' : self.seed}
        if 'svm' in self.technique:
            self._svm_parameters()
        elif 'baseline' in self.technique:
            self._baseline_parameters()
        elif 'xgb' in self.technique:
            if not hasattr(self, 'scale_pos_weight'):
                self.scale_pos_weight = 1
            if self.gpu:
                self.runtime_parameters.update(
                        {'tree_CookbookStep' : 'gpu_exact'})
            if self.hyperparameter_search:
                self.grid.update({'scale_pos_weight' :
                                  uniform(self.scale_pos_weight / 2,
                                  self.scale_pos_weight * 2)})
            else:
                self.parameters.update(
                        {'scale_pos_weight' : self.scale_pos_weight})
        return self

    def _prepare_search(self):
        self.search_options = {'random' : RandomizedSearchCV,
                               'fixed' : GridSearchCV}
#                               'bayes' : BayesSearchCV}
        self.search_runtime_parameters = {
                'estimator' : self.algorithm,
                'param_distributions' : self.grid,
                'random_state' : self.seed}
        if 'refit' in self.search_parameters:
            self.search_parameters['scoring'] = (
                    listify(self.search_parameters['scoring'])[0])
        self.search_parameters.update(self.search_runtime_parameters)
        self.search_algorithm = self.menu['cookbook']['search_algorithm']
        self.search_method = self.search_options[self.search_algorithm](
                **self.search_parameters)
        return self

    def _set_classifier(self):
        self.options = {'adaboost' : AdaBoostClassifier,
                        'baseline_classifier' : DummyClassifier,
                        'logit' : LogisticRegression,
                        'random_forest' : RandomForestClassifier,
#                        'stan' : StanModel,
                        'svm_linear' : SVC,
                        'svm_poly' : SVC,
                        'svm_rbf' : SVC,
                        'svm_sigmoid' : SVC,
#                        'tensor_flow' : KerasClassifier,
#                         torch' : NeuralNetClassifier,
                        'xgb' : XGBClassifier}
        return self

    def _set_cluster(self):
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

    def _set_defaults(self):
        getattr(self, '_set_' + self.model_type)()
        return self

    def _set_regressor(self):
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
        return self

    def _svm_parameters(self):
        svm_parameters = {'svm_linear' : 'linear',
                          'svm_poly' : 'poly',
                          'svm_rbf' : 'rbf',
                          'svm_sigmoid' : 'sigmoid'}
        self.parameters.update({'kernel' : svm_parameters[self.technique],
                                'probability' : True})
        return self

    def fit_transform(self, x, y):
        error = 'fit_transform is not implemented for machine learning models'
        raise NotImplementedError(error)
        return self

    def search(self, ingredients, recipe = None):
        if self.verbose:
            print('Searching for best hyperparameters using',
                  self.search_algorithm, 'search algorithm')
        self.search_method.fit(ingredients.x_train, ingredients.y_train)
        self.best_estimator = self.search_method.best_estimator_
        if self.verbose:
            print('The', self.search_parameters['scoring'],
                  'score of the best estimator for this', self.technique,
                  'model is', f'{self.search_method.best_score_ : 4.4f}')
        return self

    def prepare(self):
        """Adds parameters to algorithm."""
        if not hasattr(self, 'parameters') or not self.parameters:
            self.parameters = self.menu[self.technique]
        self._check_runtime_parameters()
        self._parse_parameters()
        if self.technique != 'none':
            self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def start(self, ingredients, recipe):
        if self.technique != 'none':
            if self.verbose:
                print('Applying', self.technique, 'model to data')
            if self.hyperparameter_search:
                self.search(ingredients = ingredients, recipe = recipe)
                self.algorithm = self.best_estimator
            else:
                self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return ingredients

    def transform(self, x, y):
        error = 'transform is not implemented for machine learning models'
        raise NotImplementedError(error)
        return self