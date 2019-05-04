"""
Parent and child clases for implementing the different steps in ml_funnel test
tubes.
"""
from dataclasses import dataclass
import pickle
from scipy.stats import randint, uniform
from sklearn.cluster import AffinityPropagation, Birch, KMeans
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import mutual_info_regression, RFE, RFECV
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import BayesianRidge, Lasso, LassoLars
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from category_encoders import BackwardDifferenceEncoder, BaseNEncoder
from category_encoders import BinaryEncoder, HashingEncoder, HelmertEncoder
from category_encoders import LeaveOneOutEncoder, OneHotEncoder
from category_encoders import OrdinalEncoder, SumEncoder, TargetEncoder
from category_encoders import PolynomialEncoder
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from xgboost import XGBClassifier, XGBRegressor

#from ml_funnel.logit_encode import LogitEncoder

@dataclass
class Methods(object):
    """
    Parent class for preprocessing and modeling methods in the ml_funnel.
    The Methods class allows for shared initialization, loading, and saving
    methods to be accessed by all child machine learning and preprocessing
    methods.
    """
    def __post_init__(self):
        self.settings.simplify(class_instance = self,
                               sections = ['general', 'methods'])
        return self

    def __getitem__(self, name):
        if name in self.options:
            return self.options[name]
        else:
            error_message = name + ' is not in ' + self.name + ' method'
            raise KeyError(error_message)
            return

    def __setitem__(self, name, method):
        if isinstance(name, str):
            if isinstance(method, object):
                self.options.update({name : method})
            else:
                error_message = name + ' must be a method object'
                raise TypeError(error_message)
        else:
            error_message = name + ' must be a string type'
            raise TypeError(error_message)
        return self

    def __delitem__(self, name):
        if name in self.options:
            self.options.pop(name)
        else:
            error_message = name + ' is not in ' + self.name + ' method'
            raise KeyError(error_message)
        return self

    def __contains__(self, name):
        if name in self.options:
            return True
        else:
            return False

    def _check_params(self):
        if not self.params:
            self.params = self.defaults
        return self

    def initialize(self):
        self._check_params()
        if self.runtime_params:
            self.params.update(self.runtime_params)
        if self.name != 'none':
            self.method = self.options[self.name]
            if self.params:
                self.method = self.method(**self.params)
            else:
                self.method = self.method()
        return self

    def select_params(self, params_to_use = []):
        new_params = {}
        if self.params:
            for key, value in self.params.items():
                if key in params_to_use:
                    new_params.update({key : value})
            self.params = new_params
        return self

    def apply(self, x, y = None):
        if self.name != 'none':
            self.method.fit(x, y)
            x = self.method.transform(x)
        return x

    def fit(self, x, y):
        return self.method.fit(x, y)

    def transform(self, x):
        return self.method.transform(x)

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def load(self, file_name, import_folder = '', prefix = '', suffix = ''):
        import_path = self.filer.path_join(folder = import_folder,
                                           prefix = prefix,
                                           file_name = file_name,
                                           suffix = suffix,
                                           file_type = 'pickle')
        if self.verbose:
            print('Importing', file_name)
        self.method = pickle.load(open(import_path, 'rb'))
        return self

    def save(self, file_name, export_folder = '', prefix = '', suffix = ''):
        if self.verbose:
            print('Exporting', file_name)
        export_path = self.filer.path_join(folder = export_folder,
                                           prefix = prefix,
                                           file_name = file_name,
                                           suffix = suffix,
                                           file_type = 'pickle')
        pickle.dump(self.method, open(export_path, 'wb'))
        return self

@dataclass
class Scaler(Methods):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'maxabs' : MaxAbsScaler,
                        'minmax' : MinMaxScaler,
                        'normalizer' : Normalizer,
                        'quantile' : QuantileTransformer,
                        'robust' : RobustScaler,
                        'standard' : StandardScaler}
        self.defaults = {'copy' : False}
        self.runtime_params = {}
        self.initialize()
        return self

@dataclass
class Splitter(Methods):

    name : str = ''
    params : object = None
    create_val_set : bool = False

    def __post_init__(self):
        super().__post_init__()
        self.options = {'train_test' : self._split_data,
                        'train_test_val' : self._split_data,
                        'cv' : self._cv_split,
                        'none' : self._no_split}
        self.defaults = {'test_size' : 0.33,
                         'val_size' : 0,
                         'kfolds' : 5,
                         'krepeats' : 10}
        self._check_params()
        self.method = self.options[self.name]
        self.params.update({'random_state' : self.seed})
        return self

    def _cv_split(self, data):
#        for train_index, test_index in self.params['folder'].split(data.x, data.y):
#            data.x_train, data.x_test = (data.x.iloc[train_index],
#                                         data.x.iloc[test_index])
#            data.y_train, data.y_test = (data.y.iloc[train_index],
#                                         data.y.iloc[test_index])
        return data

    def _split_data(self, data):
        data.x_train, data.x_test, data.y_train, data.y_test = (
                self._one_split(data.x, data.y, self.params['test_size']))
        if self.create_val_set:
            data.x_train, data.x_val, data.y_train, data.y_val = (
                self._one_split(data.x_train, data.y_train,
                                self.params['val_size']))
        return data

    def _one_split(self, x, y, split_size):
        x_train, x_test, y_train, y_test = (
                train_test_split(x, y,
                                 random_state = self.seed,
                                 test_size = split_size))
        return x_train, x_test, y_train, y_test

    def _no_split(self, data):
        return data

    def apply(self, data):
        return self.method(data)

    def fit(self, data):
        return self

    def transform(self, data):
        data = self.apply(data)
        return data

@dataclass
class Encoder(Methods):

    name : str = ''
    params : object = None
    columns : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'backward' : BackwardDifferenceEncoder,
                        'basen' : BaseNEncoder,
                        'binary' : BinaryEncoder,
                        'hashing' : HashingEncoder,
                        'helmert' : HelmertEncoder,
#                        'logit' : LogitEncoder,
                        'loo' : LeaveOneOutEncoder,
                        'dummy' : OneHotEncoder,
                        'ordinal' : OrdinalEncoder,
                        'sum' : SumEncoder,
                        'target' : TargetEncoder}
        self.defaults = {}
        self.runtime_params = {'cols' : self.columns}
        self.initialize()
        return self

    def fit(self, x, y):
        return self.method.fit(x, y)

    def transform(self, x):
        x = self.method.transform(x)
        for col in self.columns:
            x[col] = x[col].astype(float, copy = False)
        return x

@dataclass
class Interactor(Methods):

    name : str = ''
    params : object = None
    columns : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'polynomial' : PolynomialEncoder,
                        'quotient' : self.quotient_features,
                        'sum' : self.sum_features,
                        'difference' : self.difference_features}
        self.defaults = {}
        self.runtime_params = {'cols' : self.columns}
        self.initialize()
        return self

    def quotient_features(self):
        pass
        return self

    def sum_features(self):
        pass
        return self

    def difference_features(self):
        pass
        return self

@dataclass
class Splicer(Methods):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {}
        self.method = self.splice()
        return self

    def __getitem__(self, value):
        """
        If user wants to test different combinations of features ("splices"),
        this method returns a list of possible splicers set by user.
        """
        if self.options:
            if self.params['include_all']:
                test_columns = []
                for group, columns in self.options.items():
                    test_columns.extend(columns)
                self.options.update({'all' : test_columns})
            splicers = list(self.data.splice_options.keys())
        else:
            splicers = ['none']
        return splicers

    def splice(self):
        return self

    def add_splice(self, splice, prefixes = [], columns = []):
        """
        For the splicers in ml_funnel, this method alows users to manually
        add a new splice group to the splicer dictionary.
        """
        temp_list = self.data.create_column_list(prefixes = prefixes,
                                                 cols = columns)
        self.options.update({splice : temp_list})
        return self


    def fit(self, x, y):
        if self.params['include_all']:
            test_columns = []
            for group, columns in self.options.items():
                test_columns.extend(columns)
            self.options.update({'all' : test_columns})
        return self

    def transform(self, x):
        drop_list = [i for i in self.test_columns if i not in self.method]
        for col in drop_list:
            if col in x.columns:
                x.drop(col, axis = 'columns', inplace = True)
        return x

@dataclass
class Sampler(Methods):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'adasyn' : ADASYN,
                        'smote' : SMOTE,
                        'smoteenn' :  SMOTEENN,
                        'smotetomek' : SMOTETomek}
        self.defaults = {}
        self.runtime_params = {'random_state' : self.seed}
        self.initialize()
        return self

@dataclass
class Custom(Methods):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {}
        self.defaults = {}
        self.runtime_params = {}
        self.initialize()
        return self

@dataclass
class Selector(Methods):

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
            self.runtime_params = {'estimator' : self.model.method}
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
            self.runtime_params = {'estimator' : self.model.method}
        self.select_params(params_to_use = self.defaults.keys())
        return self

    def transform(self, x):
        if len(x.columns) > self.params['n_features_to_select']:
            return self.method.transform(x)
        else:
            return x

@dataclass
class Model(Methods):

    name : str = ''
    algorithm_type : str = ''
    params : object = None
    use_gpu : bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.algorithm_type == 'classifier':
            self.options = {'ada' : AdaBoostClassifier,
                            'logit' : LogisticRegression,
                            'random_forest' : RandomForestClassifier,
                            'svm_linear' : LinearSVC,
                            'svm' : SVC,
                            'xgb' : XGBClassifier}
        elif self.algorithm_type == 'regressor':
            self.options = {'ada' : AdaBoostRegressor,
                            'bayes_ridge' : BayesianRidge,
                            'lasso' : Lasso,
                            'lasso_lars' : LassoLars,
                            'ols' : LinearRegression,
                            'random_forest' : RandomForestRegressor,
                            'ridge' : Ridge,
                            'xgb' : XGBRegressor}
        elif self.algorithm_type == 'grouper':
            self.options = {'affinity' : AffinityPropagation,
                            'birch' : Birch,
                            'kmeans' : KMeans}
        self._parse_params()
        self.initialize()
        if self.hyperparameter_search:
            self._setup_search()
        return self

    def _setup_search(self):
        self.search_options = {'random' : RandomizedSearchCV,
                               'fixed' : GridSearchCV,
                               'bayes' : 'none'}
        self.search_runtime_params = {'estimator' : self.method,
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
                if isinstance(values[0], float):
                    self.grid.update({param : uniform(values[0], values[1])})
                elif isinstance(values[0], int):
                    self.grid.update({param : randint(values[0], values[1])})
            else:
                new_params.update({param : values})
        self.params = new_params
        self.runtime_params = {'random_state' : self.seed}
        if self.name == 'xgb':
            if not hasattr(self, 'scale_pos_weight'):
                self.scale_pos_weight = 1
            if self.use_gpu:
                self.runtime_params.update({'tree_method' : 'gpu_exact'})
            if self.use_grid:
                self.grid.update({'scale_pos_weight' :
                                  uniform(self.scale_pos_weight / 1.5,
                                  self.scale_pos_weight * 1.5)})
            else:
                self.params.update(
                        {'scale_pos_weight' : self.scale_pos_weight})
        return self

    def search(self, x, y):
        if self.verbose:
            print('Searching for best hyperparameters for the',
                  self.name, 'model using', self.search_algorithm,
                  'search method')
        self.search_method.fit(x, y)
        self.best = self.search_method.best_estimator_
        print('The', self.search_params['scoring'],
              'score of the best estimator for the', self.name,
              'model is', str(self.search_method.best_score_))
        return self