"""
analyst.components:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from types import ModuleType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import simplify
from . import base


@dataclasses.dataclass
class AnalystComponent(sourdough.project.Component):
    """Base class for parts of a data science project workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a SimpleSettings instance, 
            'name' should match the appropriate section name in a SimpleSettings 
            instance. Defaults to None. 
        contents (Any): stored item(s) for use by a Component subclass instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            False.
        after_split (ClassVar[bool]): whether the instance's method should
            only be called after the data is split into training and testing
            sets. Defaults to False.
        before_split (ClassVar[bool]): whether the instance's method should
            only be called before the data is split into training and testing
            sets. Defaults to False.
        model_limts (ClassVar[bool]): any model types that the method must be
            used with. If None are listed, simplify assumes that the instance is
            compatible with all model types. Defaults to an empty list.
                
    """
    name: str = None
    contents: Any = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    parallel: ClassVar[bool] = False
    after_split: ClassVar[bool] = False
    before_split: ClassVar[bool] = False
    model_limits: ClassVar[Sequence[str]] = []
    
    """ Public Methods """
    
    def implement(self, data: simplify.core.Dataset, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
            
        """
        if data.split and self.before_split:
            raise ValueError(
                f'{self.name} component can only be used with unsplit data')
        elif not data.split and self.after_split:
            raise ValueError(
                f'{self.name} component can only be used with split data')            
        if self.parameters:
            parameters = self.parameters
            parameters.update(kwargs)
        else:
            parameters = kwargs
        if self.contents not in [None, 'None', 'none']:
            data = self.contents.implement(data = data, **parameters)
        return data

@dataclasses.dataclass
class SimpleFill(base.SimpleStep):
    
    name: str = 'fill'
    parameters: Dict[str, Any] = dataclasses.field(default_factory = lambda: {
        'boolean': False,
        'float': 0.0,
        'integer': 0,
        'string': '',
        'categorical': '',
        'list': [],
        'datetime': 1/1/1900,
        'timedelta': 0})


@dataclasses.dataclass
class SimpleCategorize(base.SimpleStep):
    
    name: str = 'categorize'


@dataclasses.dataclass
class SimpleScale(base.SimpleStep):
    
    name: str = 'scale'


@dataclasses.dataclass
class SimpleSplit(base.SimpleStep):
    
    name: str = 'split'


@dataclasses.dataclass
class SimpleEncode(base.SimpleStep):
    
    name: str = 'encode'


@dataclasses.dataclass
class SimpleMix(base.SimpleStep):
    
    name: str = 'mix'


@dataclasses.dataclass
class SimpleCleave(base.SimpleStep):
    
    name: str = 'cleave'


@dataclasses.dataclass
class SimpleSample(base.SimpleStep):
    
    name: str = 'sample'


@dataclasses.dataclass
class SimpleReduce(base.SimpleStep):
    
    name: str = 'reduce'


@dataclasses.dataclass
class SimpleModel(base.SimpleStep):
    
    name: str = 'model'


@dataclasses.dataclass
class SimpleKNNImputer(base.SimpleTechnique):
    
    name: str = 'knn_imputer'
    module: str = 'self'
    contents: str = 'knn_impute'


@dataclasses.dataclass
class SimpleAutomaticCategorizor(base.SimpleTechnique):
    
    name: str = 'automatic_categorizer'
    module: str = 'self'
    contents: str = 'auto_categorize'


@dataclasses.dataclass
class SimpleMaxAbs(base.SimpleTechnique):
    
    name: str = 'maximum_absolute_value_scaler'
    module: str = 'sklearn.preprocessing'
    contents: str = 'MaxAbsScaler'
    default: Dict[str, Any] = dataclasses.field(default_factory = lambda: 
        {'copy': False})


@dataclasses.dataclass
class SimpleKfold(base.SimpleTechnique):
    
    name: str = 'Kfold_splitter'
    module: str = 'sklearn.model_selection'
    contents: str = 'KFold'
    default: Dict[str, Any] = dataclasses.field(default_factory = lambda: 
        {'n_splits': 5, 
         'shuffle': False})


@dataclasses.dataclass
class SimpleKfold(base.SimpleTechnique):
    
    name: str = 'Kfold_splitter'
    module: str = 'sklearn.model_selection'
    contents: str = 'KFold'
    default: Dict[str, Any] = dataclasses.field(default_factory = lambda: 
        {'n_splits': 5, 
         'shuffle': False})


@dataclasses.dataclass
class SimpleXGBoost(base.SimpleTechnique):

    name: str = 'xgboost'
    module: str = 'xgboost'
    contents: str = 'XGBClassifier'
       

# raw_options: Dict[str, simplify.SimpleTechnique] = {
#     'fill': {
#         'defaults': simplify.SimpleTechnique(
#             name = 'defaults',
#             module = 'simplify.analyst.algorithms',
#             algorithm = 'smart_fill',
#             default = {'defaults': {
#                 'boolean': False,
#                 'float': 0.0,
#                 'integer': 0,
#                 'string': '',
#                 'categorical': '',
#                 'list': [],
#                 'datetime': 1/1/1900,
#                 'timedelta': 0}}),
#         'impute': simplify.SimpleTechnique(
#             name = 'defaults',
#             module = 'sklearn.impute',
#             algorithm = 'SimpleImputer',
#             default = {'defaults': {}}),
#         'knn_impute': simplify.SimpleTechnique(
#             name = 'defaults',
#             module = 'sklearn.impute',
#             algorithm = 'KNNImputer',
#             default = {'defaults': {}})},
#     'categorize': {
#         'automatic': simplify.SimpleTechnique(
#             name = 'automatic',
#             module = 'simplify.analyst.algorithms',
#             algorithm = 'auto_categorize',
#             default = {'threshold': 10}),
#         'binary': simplify.SimpleTechnique(
#             name = 'binary',
#             module = 'sklearn.preprocessing',
#             algorithm = 'Binarizer',
#             default = {'threshold': 0.5}),
#         'bins': simplify.SimpleTechnique(
#             name = 'bins',
#             module = 'sklearn.preprocessing',
#             algorithm = 'KBinsDiscretizer',
#             default = {
#                 'strategy': 'uniform',
#                 'n_bins': 5},
#             selected = True,
#             required = {'encode': 'onehot'})},
#     'scale': {
#         'gauss': simplify.SimpleTechnique(
#             name = 'gauss',
#             module = None,
#             algorithm = 'Gaussify',
#             default = {'standardize': False, 'copy': False},
#             selected = True,
#             required = {'rescaler': 'standard'}),
#         'maxabs': simplify.SimpleTechnique(
#             name = 'maxabs',
#             module = 'sklearn.preprocessing',
#             algorithm = 'MaxAbsScaler',
#             default = {'copy': False},
#             selected = True),
#         'minmax': simplify.SimpleTechnique(
#             name = 'minmax',
#             module = 'sklearn.preprocessing',
#             algorithm = 'MinMaxScaler',
#             default = {'copy': False},
#             selected = True),
#         'normalize': simplify.SimpleTechnique(
#             name = 'normalize',
#             module = 'sklearn.preprocessing',
#             algorithm = 'Normalizer',
#             default = {'copy': False},
#             selected = True),
#         'quantile': simplify.SimpleTechnique(
#             name = 'quantile',
#             module = 'sklearn.preprocessing',
#             algorithm = 'QuantileTransformer',
#             default = {'copy': False},
#             selected = True),
#         'robust': simplify.SimpleTechnique(
#             name = 'robust',
#             module = 'sklearn.preprocessing',
#             algorithm = 'RobustScaler',
#             default = {'copy': False},
#             selected = True),
#         'standard': simplify.SimpleTechnique(
#             name = 'standard',
#             module = 'sklearn.preprocessing',
#             algorithm = 'StandardScaler',
#             default = {'copy': False},
#             selected = True)},
#     'split': {
#         'group_kfold': simplify.SimpleTechnique(
#             name = 'group_kfold',
#             module = 'sklearn.model_selection',
#             algorithm = 'GroupKFold',
#             default = {'n_splits': 5},
#             runtime = {'random_state': 'seed'},
#             selected = True,
#             fit_method = None,
#             transform_method = 'split'),
#         'kfold': simplify.SimpleTechnique(
#             name = 'kfold',
#             module = 'sklearn.model_selection',
#             algorithm = 'KFold',
#             default = {'n_splits': 5, 'shuffle': False},
#             runtime = {'random_state': 'seed'},
#             selected = True,
#             required = {'shuffle': True},
#             fit_method = None,
#             transform_method = 'split'),
#         'stratified': simplify.SimpleTechnique(
#             name = 'stratified',
#             module = 'sklearn.model_selection',
#             algorithm = 'StratifiedKFold',
#             default = {'n_splits': 5, 'shuffle': False},
#             runtime = {'random_state': 'seed'},
#             selected = True,
#             required = {'shuffle': True},
#             fit_method = None,
#             transform_method = 'split'),
#         'time': simplify.SimpleTechnique(
#             name = 'time',
#             module = 'sklearn.model_selection',
#             algorithm = 'TimeSeriesSplit',
#             default = {'n_splits': 5},
#             runtime = {'random_state': 'seed'},
#             selected = True,
#             fit_method = None,
#             transform_method = 'split'),
#         'train_test': simplify.SimpleTechnique(
#             name = 'train_test',
#             module = 'sklearn.model_selection',
#             algorithm = 'ShuffleSplit',
#             default = {'test_size': 0.33},
#             runtime = {'random_state': 'seed'},
#             required = {'n_splits': 1},
#             selected = True,
#             fit_method = None,
#             transform_method = 'split')},
#     'encode': {
#         'backward': simplify.SimpleTechnique(
#             name = 'backward',
#             module = 'category_encoders',
#             algorithm = 'BackwardDifferenceEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'basen': simplify.SimpleTechnique(
#             name = 'basen',
#             module = 'category_encoders',
#             algorithm = 'BaseNEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'binary': simplify.SimpleTechnique(
#             name = 'binary',
#             module = 'category_encoders',
#             algorithm = 'BinaryEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'dummy': simplify.SimpleTechnique(
#             name = 'dummy',
#             module = 'category_encoders',
#             algorithm = 'OneHotEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'hashing': simplify.SimpleTechnique(
#             name = 'hashing',
#             module = 'category_encoders',
#             algorithm = 'HashingEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'helmert': simplify.SimpleTechnique(
#             name = 'helmert',
#             module = 'category_encoders',
#             algorithm = 'HelmertEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'james_stein': simplify.SimpleTechnique(
#             name = 'james_stein',
#             module = 'category_encoders',
#             algorithm = 'JamesSteinEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'loo': simplify.SimpleTechnique(
#             name = 'loo',
#             module = 'category_encoders',
#             algorithm = 'LeaveOneOutEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'm_estimate': simplify.SimpleTechnique(
#             name = 'm_estimate',
#             module = 'category_encoders',
#             algorithm = 'MEstimateEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'ordinal': simplify.SimpleTechnique(
#             name = 'ordinal',
#             module = 'category_encoders',
#             algorithm = 'OrdinalEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'polynomial': simplify.SimpleTechnique(
#             name = 'polynomial_encoder',
#             module = 'category_encoders',
#             algorithm = 'PolynomialEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'sum': simplify.SimpleTechnique(
#             name = 'sum',
#             module = 'category_encoders',
#             algorithm = 'SumEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'target': simplify.SimpleTechnique(
#             name = 'target',
#             module = 'category_encoders',
#             algorithm = 'TargetEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'woe': simplify.SimpleTechnique(
#             name = 'weight_of_evidence',
#             module = 'category_encoders',
#             algorithm = 'WOEEncoder',
#             data_dependent = {'cols': 'categoricals'})},
#     'mix': {
#         'polynomial': simplify.SimpleTechnique(
#             name = 'polynomial_mixer',
#             module = 'sklearn.preprocessing',
#             algorithm = 'PolynomialFeatures',
#             default = {
#                 'degree': 2,
#                 'interaction_only': True,
#                 'include_bias': True}),
#         'quotient': simplify.SimpleTechnique(
#             name = 'quotient',
#             module = None,
#             algorithm = 'QuotientFeatures'),
#         'sum': simplify.SimpleTechnique(
#             name = 'sum',
#             module = None,
#             algorithm = 'SumFeatures'),
#         'difference': simplify.SimpleTechnique(
#             name = 'difference',
#             module = None,
#             algorithm = 'DifferenceFeatures')},
#     'cleave': {
#         'cleaver': simplify.SimpleTechnique(
#             name = 'cleaver',
#             module = 'simplify.analyst.algorithms',
#             algorithm = 'Cleaver')},
#     'sample': {
#         'adasyn': simplify.SimpleTechnique(
#             name = 'adasyn',
#             module = 'imblearn.over_sampling',
#             algorithm = 'ADASYN',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'cluster': simplify.SimpleTechnique(
#             name = 'cluster',
#             module = 'imblearn.under_sampling',
#             algorithm = 'ClusterCentroids',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'knn': simplify.SimpleTechnique(
#             name = 'knn',
#             module = 'imblearn.under_sampling',
#             algorithm = 'AllKNN',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'near_miss': simplify.SimpleTechnique(
#             name = 'near_miss',
#             module = 'imblearn.under_sampling',
#             algorithm = 'NearMiss',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'random_over': simplify.SimpleTechnique(
#             name = 'random_over',
#             module = 'imblearn.over_sampling',
#             algorithm = 'RandomOverSampler',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'random_under': simplify.SimpleTechnique(
#             name = 'random_under',
#             module = 'imblearn.under_sampling',
#             algorithm = 'RandomUnderSampler',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'smote': simplify.SimpleTechnique(
#             name = 'smote',
#             module = 'imblearn.over_sampling',
#             algorithm = 'SMOTE',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'smotenc': simplify.SimpleTechnique(
#             name = 'smotenc',
#             module = 'imblearn.over_sampling',
#             algorithm = 'SMOTENC',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             data_dependent = {
#                 'categorical_features': 'categoricals_indices'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'smoteenn': simplify.SimpleTechnique(
#             name = 'smoteenn',
#             module = 'imblearn.combine',
#             algorithm = 'SMOTEENN',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'smotetomek': simplify.SimpleTechnique(
#             name = 'smotetomek',
#             module = 'imblearn.combine',
#             algorithm = 'SMOTETomek',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample')},
#     'reduce': {
#         'kbest': simplify.SimpleTechnique(
#             name = 'kbest',
#             module = 'sklearn.feature_selection',
#             algorithm = 'SelectKBest',
#             default = {'k': 10, 'score_func': 'f_classif'},
#             selected = True),
#         'fdr': simplify.SimpleTechnique(
#             name = 'fdr',
#             module = 'sklearn.feature_selection',
#             algorithm = 'SelectFdr',
#             default = {'alpha': 0.05, 'score_func': 'f_classif'},
#             selected = True),
#         'fpr': simplify.SimpleTechnique(
#             name = 'fpr',
#             module = 'sklearn.feature_selection',
#             algorithm = 'SelectFpr',
#             default = {'alpha': 0.05, 'score_func': 'f_classif'},
#             selected = True),
#         'custom': simplify.SimpleTechnique(
#             name = 'custom',
#             module = 'sklearn.feature_selection',
#             algorithm = 'SelectFromModel',
#             default = {'threshold': 'mean'},
#             runtime = {'estimator': 'algorithm'},
#             selected = True),
#         'rank': simplify.SimpleTechnique(
#             name = 'rank',
#             module = 'simplify.critic.rank',
#             algorithm = 'RankSelect',
#             selected = True),
#         'rfe': simplify.SimpleTechnique(
#             name = 'rfe',
#             module = 'sklearn.feature_selection',
#             algorithm = 'RFE',
#             default = {'n_features_to_select': 10, 'step': 1},
#             runtime = {'estimator': 'algorithm'},
#             selected = True),
#         'rfecv': simplify.SimpleTechnique(
#             name = 'rfecv',
#             module = 'sklearn.feature_selection',
#             algorithm = 'RFECV',
#             default = {'n_features_to_select': 10, 'step': 1},
#             runtime = {'estimator': 'algorithm'},
#             selected = True)}}

# raw_model_options: Dict[str, simplify.SimpleTechnique] = {
#     'classify': {
#         'adaboost': simplify.SimpleTechnique(
#             name = 'adaboost',
#             module = 'sklearn.ensemble',
#             algorithm = 'AdaBoostClassifier',
#             transform_method = None),
#         'baseline_classifier': simplify.SimpleTechnique(
#             name = 'baseline_classifier',
#             module = 'sklearn.dummy',
#             algorithm = 'DummyClassifier',
#             required = {'strategy': 'most_frequent'},
#             transform_method = None),
#         'logit': simplify.SimpleTechnique(
#             name = 'logit',
#             module = 'sklearn.linear_model',
#             algorithm = 'LogisticRegression',
#             transform_method = None),
#         'random_forest': simplify.SimpleTechnique(
#             name = 'random_forest',
#             module = 'sklearn.ensemble',
#             algorithm = 'RandomForestClassifier',
#             transform_method = None),
#         'svm_linear': simplify.SimpleTechnique(
#             name = 'svm_linear',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'linear', 'probability': True},
#             transform_method = None),
#         'svm_poly': simplify.SimpleTechnique(
#             name = 'svm_poly',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'poly', 'probability': True},
#             transform_method = None),
#         'svm_rbf': simplify.SimpleTechnique(
#             name = 'svm_rbf',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'rbf', 'probability': True},
#             transform_method = None),
#         'svm_sigmoid': simplify.SimpleTechnique(
#             name = 'svm_sigmoid ',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'sigmoid', 'probability': True},
#             transform_method = None),
#         'tensorflow': simplify.SimpleTechnique(
#             name = 'tensorflow',
#             module = 'tensorflow',
#             algorithm = None,
#             default = {
#                 'batch_size': 10,
#                 'epochs': 2},
#             transform_method = None),
#         'xgboost': simplify.SimpleTechnique(
#             name = 'xgboost',
#             module = 'xgboost',
#             algorithm = 'XGBClassifier',
#             # data_dependent = 'scale_pos_weight',
#             transform_method = None)},
#     'cluster': {
#         'affinity': simplify.SimpleTechnique(
#             name = 'affinity',
#             module = 'sklearn.cluster',
#             algorithm = 'AffinityPropagation',
#             transform_method = None),
#         'agglomerative': simplify.SimpleTechnique(
#             name = 'agglomerative',
#             module = 'sklearn.cluster',
#             algorithm = 'AgglomerativeClustering',
#             transform_method = None),
#         'birch': simplify.SimpleTechnique(
#             name = 'birch',
#             module = 'sklearn.cluster',
#             algorithm = 'Birch',
#             transform_method = None),
#         'dbscan': simplify.SimpleTechnique(
#             name = 'dbscan',
#             module = 'sklearn.cluster',
#             algorithm = 'DBSCAN',
#             transform_method = None),
#         'kmeans': simplify.SimpleTechnique(
#             name = 'kmeans',
#             module = 'sklearn.cluster',
#             algorithm = 'KMeans',
#             transform_method = None),
#         'mean_shift': simplify.SimpleTechnique(
#             name = 'mean_shift',
#             module = 'sklearn.cluster',
#             algorithm = 'MeanShift',
#             transform_method = None),
#         'spectral': simplify.SimpleTechnique(
#             name = 'spectral',
#             module = 'sklearn.cluster',
#             algorithm = 'SpectralClustering',
#             transform_method = None),
#         'svm_linear': simplify.SimpleTechnique(
#             name = 'svm_linear',
#             module = 'sklearn.cluster',
#             algorithm = 'OneClassSVM',
#             transform_method = None),
#         'svm_poly': simplify.SimpleTechnique(
#             name = 'svm_poly',
#             module = 'sklearn.cluster',
#             algorithm = 'OneClassSVM',
#             transform_method = None),
#         'svm_rbf': simplify.SimpleTechnique(
#             name = 'svm_rbf',
#             module = 'sklearn.cluster',
#             algorithm = 'OneClassSVM,',
#             transform_method = None),
#         'svm_sigmoid': simplify.SimpleTechnique(
#             name = 'svm_sigmoid',
#             module = 'sklearn.cluster',
#             algorithm = 'OneClassSVM',
#             transform_method = None)},
#     'regress': {
#         'adaboost': simplify.SimpleTechnique(
#             name = 'adaboost',
#             module = 'sklearn.ensemble',
#             algorithm = 'AdaBoostRegressor',
#             transform_method = None),
#         'baseline_regressor': simplify.SimpleTechnique(
#             name = 'baseline_regressor',
#             module = 'sklearn.dummy',
#             algorithm = 'DummyRegressor',
#             required = {'strategy': 'mean'},
#             transform_method = None),
#         'bayes_ridge': simplify.SimpleTechnique(
#             name = 'bayes_ridge',
#             module = 'sklearn.linear_model',
#             algorithm = 'BayesianRidge',
#             transform_method = None),
#         'lasso': simplify.SimpleTechnique(
#             name = 'lasso',
#             module = 'sklearn.linear_model',
#             algorithm = 'Lasso',
#             transform_method = None),
#         'lasso_lars': simplify.SimpleTechnique(
#             name = 'lasso_lars',
#             module = 'sklearn.linear_model',
#             algorithm = 'LassoLars',
#             transform_method = None),
#         'ols': simplify.SimpleTechnique(
#             name = 'ols',
#             module = 'sklearn.linear_model',
#             algorithm = 'LinearRegression',
#             transform_method = None),
#         'random_forest': simplify.SimpleTechnique(
#             name = 'random_forest',
#             module = 'sklearn.ensemble',
#             algorithm = 'RandomForestRegressor',
#             transform_method = None),
#         'ridge': simplify.SimpleTechnique(
#             name = 'ridge',
#             module = 'sklearn.linear_model',
#             algorithm = 'Ridge',
#             transform_method = None),
#         'svm_linear': simplify.SimpleTechnique(
#             name = 'svm_linear',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'linear', 'probability': True},
#             transform_method = None),
#         'svm_poly': simplify.SimpleTechnique(
#             name = 'svm_poly',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'poly', 'probability': True},
#             transform_method = None),
#         'svm_rbf': simplify.SimpleTechnique(
#             name = 'svm_rbf',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'rbf', 'probability': True},
#             transform_method = None),
#         'svm_sigmoid': simplify.SimpleTechnique(
#             name = 'svm_sigmoid ',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'sigmoid', 'probability': True},
#             transform_method = None),
#         'xgboost': simplify.SimpleTechnique(
#             name = 'xgboost',
#             module = 'xgboost',
#             algorithm = 'XGBRegressor',
#             # data_dependent = 'scale_pos_weight',
#             transform_method = None)}}

# raw_gpu_options: Dict[str, simplify.SimpleTechnique] = {
#     'classify': {
#         'forest_inference': simplify.SimpleTechnique(
#             name = 'forest_inference',
#             module = 'cuml',
#             algorithm = 'ForestInference',
#             transform_method = None),
#         'random_forest': simplify.SimpleTechnique(
#             name = 'random_forest',
#             module = 'cuml',
#             algorithm = 'RandomForestClassifier',
#             transform_method = None),
#         'logit': simplify.SimpleTechnique(
#             name = 'logit',
#             module = 'cuml',
#             algorithm = 'LogisticRegression',
#             transform_method = None)},
#     'cluster': {
#         'dbscan': simplify.SimpleTechnique(
#             name = 'dbscan',
#             module = 'cuml',
#             algorithm = 'DBScan',
#             transform_method = None),
#         'kmeans': simplify.SimpleTechnique(
#             name = 'kmeans',
#             module = 'cuml',
#             algorithm = 'KMeans',
#             transform_method = None)},
#     'regressor': {
#         'lasso': simplify.SimpleTechnique(
#             name = 'lasso',
#             module = 'cuml',
#             algorithm = 'Lasso',
#             transform_method = None),
#         'ols': simplify.SimpleTechnique(
#             name = 'ols',
#             module = 'cuml',
#             algorithm = 'LinearRegression',
#             transform_method = None),
#         'ridge': simplify.SimpleTechnique(
#             name = 'ridge',
#             module = 'cuml',
#             algorithm = 'RidgeRegression',
#             transform_method = None)}}

# def get_algorithms(settings: Mapping[str, Any]) -> sourdough.types.Catalog:
#     """[summary]

#     Args:
#         project (sourdough.Project): [description]

#     Returns:
#         sourdough.types.Catalog: [description]
        
#     """
#     algorithms = raw_options
#     algorithms['model'] = raw_model_options[settings['analyst']['model_type']]
#     if settings['general']['gpu']:
#         algorithms['model'].update(
#             raw_gpu_options[settings['analyst']['model_type']])
#     return algorithms
