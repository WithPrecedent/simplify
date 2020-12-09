"""
analyst: modeling and analytic classes and functions
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
import pathlib
from types import ModuleType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import numpy as np
import pandas as pd

import simplify
import sourdough


@dataclasses.dataclass
class Analyze(simplify.SimpleProject):
    """Constructs, organizes, and implements data analysis.

    Args:
        contents (Mapping[str, object]]): stored objects created by the 
            'create' methods of 'creators'. Defaults to an empty dict.
        settings (Union[Type, str, pathlib.Path]]): a Settings-compatible class,
            a str or pathlib.Path containing the file path where a file of a 
            supported file type with settings for a Settings instance is 
            located. Defaults to the default Settings instance.
        manager (Union[Type, str, pathlib.Path]]): a Manager-compatible class,
            or a str or pathlib.Path containing the full path of where the root 
            folder should be located for file input and output. A 'manager'
            must contain all file path and import/export methods for use 
            throughout sourdough. Defaults to the default Manager instance. 
        creators (Sequence[Union[Type, str]]): a Creator-compatible classes or
            strings corresponding to the keys in registry of the default
            'creator' in 'bases'. Defaults to a list of 'simple_architect', 
            'simple_builder', and 'simple_worker'. 
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example if a 
            sourdough instance needs settings from a Settings instance, 'name' 
            should match the appropriate section name in the Settings instance. 
            When subclassing, it is sometimes a good idea to use the same 'name' 
            attribute as the base class for effective coordination between 
            sourdough classes. If it is None, the 'name' will be attempted to be 
            inferred from the first section name in 'settings' after 'general' 
            and 'files'. If that fails, 'name' will be the snakecase name of the
            class. Defaults to None. 
        identification (str): a unique identification name for a Project 
            instance. The name is used for creating file folders related to the 
            project. If it is None, a str will be created from 'name' and the 
            date and time. Defaults to None.   
        automatic (bool): whether to automatically advance 'director' (True) or 
            whether the director must be advanced manually (False). Defaults to 
            True.
        data (object): any data object for the project to be applied. If it is
            None, an instance will still execute its workflow, but it won't
            apply it to any external data. Defaults to None.  
        bases (ClassVar[object]): contains information about default base 
            classes used by a Project instance. Defaults to an instance of 
            SimpleBases.

    """
    contents: Sequence[Any] = dataclasses.field(default_factory = dict)
    settings: Union[object, Type, str, pathlib.Path] = None
    manager: Union[object, Type, str, pathlib.Path] = None
    creators: Sequence[Union[Type, str]] = dataclasses.field(
        default_factory = lambda: ['analyst_architect', 'analyst_builder', 
                                   'analyst_worker'])
    name: str = None
    identification: str = None
    automatic: bool = True
    data: Union[pd.DataFrame, np.ndArray, simplify.Dataset] = None
    bases: ClassVar[object] = simplify.SimpleBases()

    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        sourdough.rules.validations.append('data')
        # Calls parent and/or mixin initialization method(s).
        try:
            super().__post_init__()
        except AttributeError:
            pass
        
    """ Private Methods """
    
    def _validate_data(self) -> None:
        """Validates 'data' or converts it to a Dataset instance."""
        pass



@dataclasses.dataclass
class AnalystAlgorithm(simplify.Algorithm):
    
    pass



raw_options: Dict[str, AnalystAlgorithm] = {
    'fill': {
        'defaults': AnalystAlgorithm(
            name = 'defaults',
            module = 'simplify.analyst.algorithms',
            algorithm = 'smart_fill',
            default = {'defaults': {
                'boolean': False,
                'float': 0.0,
                'integer': 0,
                'string': '',
                'categorical': '',
                'list': [],
                'datetime': 1/1/1900,
                'timedelta': 0}}),
        'impute': AnalystAlgorithm(
            name = 'defaults',
            module = 'sklearn.impute',
            algorithm = 'SimpleImputer',
            default = {'defaults': {}}),
        'knn_impute': AnalystAlgorithm(
            name = 'defaults',
            module = 'sklearn.impute',
            algorithm = 'KNNImputer',
            default = {'defaults': {}})},
    'categorize': {
        'automatic': AnalystAlgorithm(
            name = 'automatic',
            module = 'simplify.analyst.algorithms',
            algorithm = 'auto_categorize',
            default = {'threshold': 10}),
        'binary': AnalystAlgorithm(
            name = 'binary',
            module = 'sklearn.preprocessing',
            algorithm = 'Binarizer',
            default = {'threshold': 0.5}),
        'bins': AnalystAlgorithm(
            name = 'bins',
            module = 'sklearn.preprocessing',
            algorithm = 'KBinsDiscretizer',
            default = {
                'strategy': 'uniform',
                'n_bins': 5},
            selected = True,
            required = {'encode': 'onehot'})},
    'scale': {
        'gauss': AnalystAlgorithm(
            name = 'gauss',
            module = None,
            algorithm = 'Gaussify',
            default = {'standardize': False, 'copy': False},
            selected = True,
            required = {'rescaler': 'standard'}),
        'maxabs': AnalystAlgorithm(
            name = 'maxabs',
            module = 'sklearn.preprocessing',
            algorithm = 'MaxAbsScaler',
            default = {'copy': False},
            selected = True),
        'minmax': AnalystAlgorithm(
            name = 'minmax',
            module = 'sklearn.preprocessing',
            algorithm = 'MinMaxScaler',
            default = {'copy': False},
            selected = True),
        'normalize': AnalystAlgorithm(
            name = 'normalize',
            module = 'sklearn.preprocessing',
            algorithm = 'Normalizer',
            default = {'copy': False},
            selected = True),
        'quantile': AnalystAlgorithm(
            name = 'quantile',
            module = 'sklearn.preprocessing',
            algorithm = 'QuantileTransformer',
            default = {'copy': False},
            selected = True),
        'robust': AnalystAlgorithm(
            name = 'robust',
            module = 'sklearn.preprocessing',
            algorithm = 'RobustScaler',
            default = {'copy': False},
            selected = True),
        'standard': AnalystAlgorithm(
            name = 'standard',
            module = 'sklearn.preprocessing',
            algorithm = 'StandardScaler',
            default = {'copy': False},
            selected = True)},
    'split': {
        'group_kfold': AnalystAlgorithm(
            name = 'group_kfold',
            module = 'sklearn.model_selection',
            algorithm = 'GroupKFold',
            default = {'n_splits': 5},
            runtime = {'random_state': 'seed'},
            selected = True,
            fit_method = None,
            transform_method = 'split'),
        'kfold': AnalystAlgorithm(
            name = 'kfold',
            module = 'sklearn.model_selection',
            algorithm = 'KFold',
            default = {'n_splits': 5, 'shuffle': False},
            runtime = {'random_state': 'seed'},
            selected = True,
            required = {'shuffle': True},
            fit_method = None,
            transform_method = 'split'),
        'stratified': AnalystAlgorithm(
            name = 'stratified',
            module = 'sklearn.model_selection',
            algorithm = 'StratifiedKFold',
            default = {'n_splits': 5, 'shuffle': False},
            runtime = {'random_state': 'seed'},
            selected = True,
            required = {'shuffle': True},
            fit_method = None,
            transform_method = 'split'),
        'time': AnalystAlgorithm(
            name = 'time',
            module = 'sklearn.model_selection',
            algorithm = 'TimeSeriesSplit',
            default = {'n_splits': 5},
            runtime = {'random_state': 'seed'},
            selected = True,
            fit_method = None,
            transform_method = 'split'),
        'train_test': AnalystAlgorithm(
            name = 'train_test',
            module = 'sklearn.model_selection',
            algorithm = 'ShuffleSplit',
            default = {'test_size': 0.33},
            runtime = {'random_state': 'seed'},
            required = {'n_splits': 1},
            selected = True,
            fit_method = None,
            transform_method = 'split')},
    'encode': {
        'backward': AnalystAlgorithm(
            name = 'backward',
            module = 'category_encoders',
            algorithm = 'BackwardDifferenceEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'basen': AnalystAlgorithm(
            name = 'basen',
            module = 'category_encoders',
            algorithm = 'BaseNEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'binary': AnalystAlgorithm(
            name = 'binary',
            module = 'category_encoders',
            algorithm = 'BinaryEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'dummy': AnalystAlgorithm(
            name = 'dummy',
            module = 'category_encoders',
            algorithm = 'OneHotEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'hashing': AnalystAlgorithm(
            name = 'hashing',
            module = 'category_encoders',
            algorithm = 'HashingEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'helmert': AnalystAlgorithm(
            name = 'helmert',
            module = 'category_encoders',
            algorithm = 'HelmertEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'james_stein': AnalystAlgorithm(
            name = 'james_stein',
            module = 'category_encoders',
            algorithm = 'JamesSteinEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'loo': AnalystAlgorithm(
            name = 'loo',
            module = 'category_encoders',
            algorithm = 'LeaveOneOutEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'm_estimate': AnalystAlgorithm(
            name = 'm_estimate',
            module = 'category_encoders',
            algorithm = 'MEstimateEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'ordinal': AnalystAlgorithm(
            name = 'ordinal',
            module = 'category_encoders',
            algorithm = 'OrdinalEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'polynomial': AnalystAlgorithm(
            name = 'polynomial_encoder',
            module = 'category_encoders',
            algorithm = 'PolynomialEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'sum': AnalystAlgorithm(
            name = 'sum',
            module = 'category_encoders',
            algorithm = 'SumEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'target': AnalystAlgorithm(
            name = 'target',
            module = 'category_encoders',
            algorithm = 'TargetEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'woe': AnalystAlgorithm(
            name = 'weight_of_evidence',
            module = 'category_encoders',
            algorithm = 'WOEEncoder',
            data_dependent = {'cols': 'categoricals'})},
    'mix': {
        'polynomial': AnalystAlgorithm(
            name = 'polynomial_mixer',
            module = 'sklearn.preprocessing',
            algorithm = 'PolynomialFeatures',
            default = {
                'degree': 2,
                'interaction_only': True,
                'include_bias': True}),
        'quotient': AnalystAlgorithm(
            name = 'quotient',
            module = None,
            algorithm = 'QuotientFeatures'),
        'sum': AnalystAlgorithm(
            name = 'sum',
            module = None,
            algorithm = 'SumFeatures'),
        'difference': AnalystAlgorithm(
            name = 'difference',
            module = None,
            algorithm = 'DifferenceFeatures')},
    'cleave': {
        'cleaver': AnalystAlgorithm(
            name = 'cleaver',
            module = 'simplify.analyst.algorithms',
            algorithm = 'Cleaver')},
    'sample': {
        'adasyn': AnalystAlgorithm(
            name = 'adasyn',
            module = 'imblearn.over_sampling',
            algorithm = 'ADASYN',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            fit_method = None,
            transform_method = 'fit_resample'),
        'cluster': AnalystAlgorithm(
            name = 'cluster',
            module = 'imblearn.under_sampling',
            algorithm = 'ClusterCentroids',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            fit_method = None,
            transform_method = 'fit_resample'),
        'knn': AnalystAlgorithm(
            name = 'knn',
            module = 'imblearn.under_sampling',
            algorithm = 'AllKNN',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            fit_method = None,
            transform_method = 'fit_resample'),
        'near_miss': AnalystAlgorithm(
            name = 'near_miss',
            module = 'imblearn.under_sampling',
            algorithm = 'NearMiss',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            fit_method = None,
            transform_method = 'fit_resample'),
        'random_over': AnalystAlgorithm(
            name = 'random_over',
            module = 'imblearn.over_sampling',
            algorithm = 'RandomOverSampler',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            fit_method = None,
            transform_method = 'fit_resample'),
        'random_under': AnalystAlgorithm(
            name = 'random_under',
            module = 'imblearn.under_sampling',
            algorithm = 'RandomUnderSampler',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            fit_method = None,
            transform_method = 'fit_resample'),
        'smote': AnalystAlgorithm(
            name = 'smote',
            module = 'imblearn.over_sampling',
            algorithm = 'SMOTE',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            fit_method = None,
            transform_method = 'fit_resample'),
        'smotenc': AnalystAlgorithm(
            name = 'smotenc',
            module = 'imblearn.over_sampling',
            algorithm = 'SMOTENC',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            data_dependent = {
                'categorical_features': 'categoricals_indices'},
            fit_method = None,
            transform_method = 'fit_resample'),
        'smoteenn': AnalystAlgorithm(
            name = 'smoteenn',
            module = 'imblearn.combine',
            algorithm = 'SMOTEENN',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            fit_method = None,
            transform_method = 'fit_resample'),
        'smotetomek': AnalystAlgorithm(
            name = 'smotetomek',
            module = 'imblearn.combine',
            algorithm = 'SMOTETomek',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            fit_method = None,
            transform_method = 'fit_resample')},
    'reduce': {
        'kbest': AnalystAlgorithm(
            name = 'kbest',
            module = 'sklearn.feature_selection',
            algorithm = 'SelectKBest',
            default = {'k': 10, 'score_func': 'f_classif'},
            selected = True),
        'fdr': AnalystAlgorithm(
            name = 'fdr',
            module = 'sklearn.feature_selection',
            algorithm = 'SelectFdr',
            default = {'alpha': 0.05, 'score_func': 'f_classif'},
            selected = True),
        'fpr': AnalystAlgorithm(
            name = 'fpr',
            module = 'sklearn.feature_selection',
            algorithm = 'SelectFpr',
            default = {'alpha': 0.05, 'score_func': 'f_classif'},
            selected = True),
        'custom': AnalystAlgorithm(
            name = 'custom',
            module = 'sklearn.feature_selection',
            algorithm = 'SelectFromModel',
            default = {'threshold': 'mean'},
            runtime = {'estimator': 'algorithm'},
            selected = True),
        'rank': AnalystAlgorithm(
            name = 'rank',
            module = 'simplify.critic.rank',
            algorithm = 'RankSelect',
            selected = True),
        'rfe': AnalystAlgorithm(
            name = 'rfe',
            module = 'sklearn.feature_selection',
            algorithm = 'RFE',
            default = {'n_features_to_select': 10, 'step': 1},
            runtime = {'estimator': 'algorithm'},
            selected = True),
        'rfecv': AnalystAlgorithm(
            name = 'rfecv',
            module = 'sklearn.feature_selection',
            algorithm = 'RFECV',
            default = {'n_features_to_select': 10, 'step': 1},
            runtime = {'estimator': 'algorithm'},
            selected = True)}}

raw_model_options: Dict[str, AnalystAlgorithm] = {
    'classify': {
        'adaboost': AnalystAlgorithm(
            name = 'adaboost',
            module = 'sklearn.ensemble',
            algorithm = 'AdaBoostClassifier',
            transform_method = None),
        'baseline_classifier': AnalystAlgorithm(
            name = 'baseline_classifier',
            module = 'sklearn.dummy',
            algorithm = 'DummyClassifier',
            required = {'strategy': 'most_frequent'},
            transform_method = None),
        'logit': AnalystAlgorithm(
            name = 'logit',
            module = 'sklearn.linear_model',
            algorithm = 'LogisticRegression',
            transform_method = None),
        'random_forest': AnalystAlgorithm(
            name = 'random_forest',
            module = 'sklearn.ensemble',
            algorithm = 'RandomForestClassifier',
            transform_method = None),
        'svm_linear': AnalystAlgorithm(
            name = 'svm_linear',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            required = {'kernel': 'linear', 'probability': True},
            transform_method = None),
        'svm_poly': AnalystAlgorithm(
            name = 'svm_poly',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            required = {'kernel': 'poly', 'probability': True},
            transform_method = None),
        'svm_rbf': AnalystAlgorithm(
            name = 'svm_rbf',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            required = {'kernel': 'rbf', 'probability': True},
            transform_method = None),
        'svm_sigmoid': AnalystAlgorithm(
            name = 'svm_sigmoid ',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            required = {'kernel': 'sigmoid', 'probability': True},
            transform_method = None),
        'tensorflow': AnalystAlgorithm(
            name = 'tensorflow',
            module = 'tensorflow',
            algorithm = None,
            default = {
                'batch_size': 10,
                'epochs': 2},
            transform_method = None),
        'xgboost': AnalystAlgorithm(
            name = 'xgboost',
            module = 'xgboost',
            algorithm = 'XGBClassifier',
            # data_dependent = 'scale_pos_weight',
            transform_method = None)},
    'cluster': {
        'affinity': AnalystAlgorithm(
            name = 'affinity',
            module = 'sklearn.cluster',
            algorithm = 'AffinityPropagation',
            transform_method = None),
        'agglomerative': AnalystAlgorithm(
            name = 'agglomerative',
            module = 'sklearn.cluster',
            algorithm = 'AgglomerativeClustering',
            transform_method = None),
        'birch': AnalystAlgorithm(
            name = 'birch',
            module = 'sklearn.cluster',
            algorithm = 'Birch',
            transform_method = None),
        'dbscan': AnalystAlgorithm(
            name = 'dbscan',
            module = 'sklearn.cluster',
            algorithm = 'DBSCAN',
            transform_method = None),
        'kmeans': AnalystAlgorithm(
            name = 'kmeans',
            module = 'sklearn.cluster',
            algorithm = 'KMeans',
            transform_method = None),
        'mean_shift': AnalystAlgorithm(
            name = 'mean_shift',
            module = 'sklearn.cluster',
            algorithm = 'MeanShift',
            transform_method = None),
        'spectral': AnalystAlgorithm(
            name = 'spectral',
            module = 'sklearn.cluster',
            algorithm = 'SpectralClustering',
            transform_method = None),
        'svm_linear': AnalystAlgorithm(
            name = 'svm_linear',
            module = 'sklearn.cluster',
            algorithm = 'OneClassSVM',
            transform_method = None),
        'svm_poly': AnalystAlgorithm(
            name = 'svm_poly',
            module = 'sklearn.cluster',
            algorithm = 'OneClassSVM',
            transform_method = None),
        'svm_rbf': AnalystAlgorithm(
            name = 'svm_rbf',
            module = 'sklearn.cluster',
            algorithm = 'OneClassSVM,',
            transform_method = None),
        'svm_sigmoid': AnalystAlgorithm(
            name = 'svm_sigmoid',
            module = 'sklearn.cluster',
            algorithm = 'OneClassSVM',
            transform_method = None)},
    'regress': {
        'adaboost': AnalystAlgorithm(
            name = 'adaboost',
            module = 'sklearn.ensemble',
            algorithm = 'AdaBoostRegressor',
            transform_method = None),
        'baseline_regressor': AnalystAlgorithm(
            name = 'baseline_regressor',
            module = 'sklearn.dummy',
            algorithm = 'DummyRegressor',
            required = {'strategy': 'mean'},
            transform_method = None),
        'bayes_ridge': AnalystAlgorithm(
            name = 'bayes_ridge',
            module = 'sklearn.linear_model',
            algorithm = 'BayesianRidge',
            transform_method = None),
        'lasso': AnalystAlgorithm(
            name = 'lasso',
            module = 'sklearn.linear_model',
            algorithm = 'Lasso',
            transform_method = None),
        'lasso_lars': AnalystAlgorithm(
            name = 'lasso_lars',
            module = 'sklearn.linear_model',
            algorithm = 'LassoLars',
            transform_method = None),
        'ols': AnalystAlgorithm(
            name = 'ols',
            module = 'sklearn.linear_model',
            algorithm = 'LinearRegression',
            transform_method = None),
        'random_forest': AnalystAlgorithm(
            name = 'random_forest',
            module = 'sklearn.ensemble',
            algorithm = 'RandomForestRegressor',
            transform_method = None),
        'ridge': AnalystAlgorithm(
            name = 'ridge',
            module = 'sklearn.linear_model',
            algorithm = 'Ridge',
            transform_method = None),
        'svm_linear': AnalystAlgorithm(
            name = 'svm_linear',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            required = {'kernel': 'linear', 'probability': True},
            transform_method = None),
        'svm_poly': AnalystAlgorithm(
            name = 'svm_poly',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            required = {'kernel': 'poly', 'probability': True},
            transform_method = None),
        'svm_rbf': AnalystAlgorithm(
            name = 'svm_rbf',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            required = {'kernel': 'rbf', 'probability': True},
            transform_method = None),
        'svm_sigmoid': AnalystAlgorithm(
            name = 'svm_sigmoid ',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            required = {'kernel': 'sigmoid', 'probability': True},
            transform_method = None),
        'xgboost': AnalystAlgorithm(
            name = 'xgboost',
            module = 'xgboost',
            algorithm = 'XGBRegressor',
            # data_dependent = 'scale_pos_weight',
            transform_method = None)}}

raw_gpu_options: Dict[str, AnalystAlgorithm] = {
    'classify': {
        'forest_inference': AnalystAlgorithm(
            name = 'forest_inference',
            module = 'cuml',
            algorithm = 'ForestInference',
            transform_method = None),
        'random_forest': AnalystAlgorithm(
            name = 'random_forest',
            module = 'cuml',
            algorithm = 'RandomForestClassifier',
            transform_method = None),
        'logit': AnalystAlgorithm(
            name = 'logit',
            module = 'cuml',
            algorithm = 'LogisticRegression',
            transform_method = None)},
    'cluster': {
        'dbscan': AnalystAlgorithm(
            name = 'dbscan',
            module = 'cuml',
            algorithm = 'DBScan',
            transform_method = None),
        'kmeans': AnalystAlgorithm(
            name = 'kmeans',
            module = 'cuml',
            algorithm = 'KMeans',
            transform_method = None)},
    'regressor': {
        'lasso': AnalystAlgorithm(
            name = 'lasso',
            module = 'cuml',
            algorithm = 'Lasso',
            transform_method = None),
        'ols': AnalystAlgorithm(
            name = 'ols',
            module = 'cuml',
            algorithm = 'LinearRegression',
            transform_method = None),
        'ridge': AnalystAlgorithm(
            name = 'ridge',
            module = 'cuml',
            algorithm = 'RidgeRegression',
            transform_method = None)}}


@dataclasses.dataclass
class AnalystAlgorithms(sourdough.types.Catalog):
    """A dictonary of AnalystAlgorithm options for the Analyst subpackage.

    Args:
        contents (Mapping[Any, Any]]): stored dictionary. Defaults to an empty 
            dict.
        defaults (Sequence[Any]]): a list of keys in 'contents' which will be 
            used to return items when 'default' is sought. If not passed, 
            'default' will be set to all keys.
        always_return_list (bool): whether to return a list even when the key 
            passed is not a list or special access key (True) or to return a 
            list only when a list or special access key is used (False). 
            Defaults to False.
        project
        
    """
    contents: Mapping[Any, Any] = dataclasses.field(default_factory = dict)  
    defaults: Sequence[Any] = dataclasses.field(default_factory = list)
    always_return_list: bool = False
    project: simplify.SimpleProject = None
    
    """ Initialization Methods """
    
    def __post_init__(self) -> None:
        """Initializes class instance."""
        self._create_contents()
        # Calls parent initialization methods, if they exist.
        try:
            super().__post_init__()
        except AttributeError:
            pass      
        
    """ Private Methods """
    
    def _create_contents(self) -> None:
        """Populates 'contents' based on 'project.settings'."""
        self.contents = raw_options
        self.contents['model'] = raw_model_options[
            self.project.settings['analyst']['model_type']]
        if self.project.settings['general']['gpu']:
            self.contents['model'].update(
                raw_gpu_options[self.project.settings['analyst']['model_type']])
        return self
