
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
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#from pystan import StanModel
from xgboost import XGBClassifier, XGBRegressor

from ...implements.technique import Technique
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

    def _check_runtime_parameters(self):
        """Checks if class has runtime_parameters and, if so, adds them to
        the parameters attribute.
        """
        if hasattr(self, 'runtime_parameters') and self.runtime_parameters:
            self.runtime_parameters.update({'random_state' : self.seed})
        else:
            self.runtime_parameters = {'random_state' : self.seed}
        self.parameters.update(self.runtime_parameters)
        return 
    
    def _parse_parameters(self):
        """

        :return: 
        """
        self.hyperparameter_search = False
        self.space = {}
        new_parameters = {}
        for param, values in self.parameters.items():
            if isinstance(values, list):
                self.hyperparameter_search = True
                if self._list_type(values, float):
                    self.space.update({param : uniform(values[0], values[1])})
                elif self._list_type(values, int):
                    self.space.update({param : randint(values[0], values[1])})
            else:
                new_parameters.update({param : values})
        self.parameters = new_parameters
        return self

    def _prepare_search(self):
        self.searcher = Search(
                technique = self.menu['cookbook']['search_algorithm'],
                estimator = self.algorithm,
                parameters = self.search_parameters,
                space = self.space,
                seed = self.seed,
                verbose = self.verbose)
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
                        'tensor_flow' : KerasClassifier,
#                        'torch' : NeuralNetClassifier,
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
#                        'stan' : StanModel,
                        'svm_linear' : SVR,
                        'svm_poly' : SVR,
                        'svm_rbf' : SVR,
                        'svm_sigmoid' : SVR,
                        'tensor_flow' : KerasRegressor,
#                        'torch' : NeuralNetRegressor,
                        'xgb' : XGBRegressor}
        return self

    def _specific_parameters(self):
        self.runtime_parameters = {'random_state' : self.seed}
        if hasattr(self, '_' + self.technique + '_parameters'):
            getattr(self, '_' + self.technique + '_parameters')()        
        return self
    
    def _svm_parameters(self):
        svm_parameters = {'svm_linear' : 'linear',
                          'svm_poly' : 'poly',
                          'svm_rbf' : 'rbf',
                          'svm_sigmoid' : 'sigmoid'}
        self.parameters.update({'kernel' : svm_parameters[self.technique],
                                'probability' : True})
        return self

    def _tensor_flow_model(self):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten
        classifier = Sequential()
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
            activation = 'relu', input_dim = 30))
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
            activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
            activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', 
                           loss = 'binary_crossentropy', 
                           metrics = ['accuracy'])
        return classifier
#        model = Sequential()
#        model.add(Activation('relu'))
#        model.add(Activation('relu'))
#        model.add(Dropout(0.25))   
#        model.add(Flatten())
#        for layer_size in self.parameters['dense_layer_sizes']:
#            model.add(Dense(layer_size))
#            model.add(Activation('relu'))
#        model.add(Dropout(0.5))
#        model.add(Dense(2))
#        model.add(Activation('softmax'))    
#        model.compile(loss = 'categorical_crossentropy',
#                      optimizer = 'adadelta',
#                      metrics = ['accuracy'])
#        return model        
  
    def _tensor_flow_parameters(self):
        new_parameters = {'build_fn' : self._tensor_flow_model,
                          'batch_size' : 10, 
                          'epochs' : 2}
        self.parameters = new_parameters
        return self
    
    def _xgb_parameters(self):
        if not hasattr(self, 'scale_pos_weight'):
            self.scale_pos_weight = 1
        if self.gpu:
            if self.verbose:
                print('Using GPU')
            self.runtime_parameters.update(
                    {'tree_method' : 'gpu_exact'})
        elif self.verbose:
            print('Using CPU')
        if self.hyperparameter_search:
            self.space.update({'scale_pos_weight' :
                               uniform(self.scale_pos_weight / 2,
                               self.scale_pos_weight * 2)})
        else:
            self.parameters.update(
                    {'scale_pos_weight' : self.scale_pos_weight})        
        return self
    
    def fit_transform(self, x, y):
        error = 'fit_transform is not implemented for machine learning models'
        raise NotImplementedError(error)

    def prepare(self):
        """Adds parameters to algorithm."""
        if not hasattr(self, 'parameters') or not self.parameters:
            self.parameters = self.menu[self.technique]
        self._check_runtime_parameters()
        self._parse_parameters()
        self._specific_parameters()
        if self.technique != 'none':
            self.algorithm = self.options[self.technique](**self.parameters)
        if self.hyperparameter_search:
            self._prepare_search()
        return self

    def start(self, ingredients, recipe):
        """Applies model from recipe to ingredients data."""
        if self.technique != 'none':
            if self.verbose:
                print('Applying', self.technique, 'model')
            if self.hyperparameter_search:
                self.searcher.start(ingredients = ingredients)
                self.algorithm = self.searcher.best_estimator
            else:
                self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return ingredients

    def transform(self, x, y):
        error = 'transform is not implemented for machine learning models'
        raise NotImplementedError(error)
    
@dataclass
class Search(Technique):
    
    technique : str
    estimator : object
    parameters : object
    space : object
    seed : int
    verbose : bool

    def __post_init__(self):
        super().__post_init__()        
        return self
    
    def _set_defaults(self):
        self.options = {'random' : RandomizedSearchCV,
                        'grid' : GridSearchCV}
#                       'bayes' : BayesSearchCV} 
        self.runtime_parameters = {'estimator' : self.estimator,
                                   'param_distributions' : self.space,
                                   'random_state' : self.seed}
        return self

    def prepare(self):
        if 'refit' in self.parameters:
            self.parameters['scoring'] = listify(self.parameters['scoring'])[0]
        self.parameters.update(self.runtime_parameters)
        self.tool = self.options[self.technique](**self.parameters)
        return self
    
    def start(self, ingredients):
        if self.verbose:
            print('Searching for best hyperparameters using',
                  self.technique, 'search algorithm')
        self.tool.fit(ingredients.x_train, ingredients.y_train)
        self.best_estimator = self.tool.best_estimator_
        if self.verbose:
            print('The', self.parameters['scoring'],
                  'score of the best estimator for this model is', 
                  f'{self.tool.best_score_ : 4.4f}')
        return self    