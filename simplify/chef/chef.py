"""
.. module:: chef
:synopsis: machine learning made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simplify.core.base import SimpleCatalog
from simplify.core.book import Book
from simplify.core.book import Algorithm


@dataclass
class ChefCatalog(SimpleCatalog):
    """A dictonary of Algorithm options for the Chef subpackage.

    Args:
        dictionary (Optional[str, Any]): default stored dictionary. Defaults to
            an empty dictionary.
        wildcards (Optional[List[str]]): a list of corresponding properties
            which access sets of dictionary keys. If none is passed, the two
            included properties ('default' and 'all') are used.
        defaults (Optional[List[str]]): a list of keys in 'dictionary' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        null_value (Optional[Any]): value to return when 'none' is accessed or
            an item isn't found in 'dictionary'. Defaults to None.

    """
    dictionary: Optional[Dict[str, Any]] = field(default_factory = dict)
    wildcards: Optional[List[str]] = field(default_factory = list)
    defaults: Optional[List[str]] = field(default_factory = list)
    null_value: Optional[Any] = None
    project: 'Project' = None

    def __post_init__(self) -> None:
        """Initializes 'defaults' and 'wildcards'."""
        super().__post_init__()
        if not self.dictionary:
            self._create_dictionary()
        return self

    def _create_dictionary(self) -> None:
        self.dictionary = {
            'filler': {
                'defaults': Algorithm(
                    name = 'defaults',
                    module = 'simplify.chef.algorithms',
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
                'impute': Algorithm(
                    name = 'defaults',
                    module = 'sklearn.impute',
                    algorithm = 'SimpleImputer',
                    default = {'defaults': {}}),
                'knn_impute': Algorithm(
                    name = 'defaults',
                    module = 'sklearn.impute',
                    algorithm = 'KNNImputer',
                    default = {'defaults': {}})},
            'categorizer': {
                'automatic': Algorithm(
                    name = 'automatic',
                    module = 'simplify.chef.algorithms',
                    algorithm = 'auto_categorize',
                    default = {'threshold': 10}),
                'binary': Algorithm(
                    name = 'binary',
                    module = 'sklearn.preprocessing',
                    algorithm = 'Binarizer',
                    default = {'threshold': 0.5}),
                'bins': Algorithm(
                    name = 'bins',
                    module = 'sklearn.preprocessing',
                    algorithm = 'KBinsDiscretizer',
                    default = {
                        'strategy': 'uniform',
                        'n_bins': 5},
                    selected = True,
                    required = {'encode': 'onehot'})},
            'scaler': {
                'gauss': Algorithm(
                    name = 'gauss',
                    module = None,
                    algorithm = 'Gaussify',
                    default = {'standardize': False, 'copy': False},
                    selected = True,
                    required = {'rescaler': 'standard'}),
                'maxabs': Algorithm(
                    name = 'maxabs',
                    module = 'sklearn.preprocessing',
                    algorithm = 'MaxAbsScaler',
                    default = {'copy': False},
                    selected = True),
                'minmax': Algorithm(
                    name = 'minmax',
                    module = 'sklearn.preprocessing',
                    algorithm = 'MinMaxScaler',
                    default = {'copy': False},
                    selected = True),
                'normalize': Algorithm(
                    name = 'normalize',
                    module = 'sklearn.preprocessing',
                    algorithm = 'Normalizer',
                    default = {'copy': False},
                    selected = True),
                'quantile': Algorithm(
                    name = 'quantile',
                    module = 'sklearn.preprocessing',
                    algorithm = 'QuantileTransformer',
                    default = {'copy': False},
                    selected = True),
                'robust': Algorithm(
                    name = 'robust',
                    module = 'sklearn.preprocessing',
                    algorithm = 'RobustScaler',
                    default = {'copy': False},
                    selected = True),
                'standard': Algorithm(
                    name = 'standard',
                    module = 'sklearn.preprocessing',
                    algorithm = 'StandardScaler',
                    default = {'copy': False},
                    selected = True)},
            'splitter': {
                'group_kfold': Algorithm(
                    name = 'group_kfold',
                    module = 'sklearn.model_selection',
                    algorithm = 'GroupKFold',
                    default = {'n_splits': 5},
                    runtime = {'random_state': 'seed'},
                    selected = True),
                'kfold': Algorithm(
                    name = 'kfold',
                    module = 'sklearn.model_selection',
                    algorithm = 'KFold',
                    default = {'n_splits': 5, 'shuffle': False},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    required = {'shuffle': True}),
                'stratified': Algorithm(
                    name = 'stratified',
                    module = 'sklearn.model_selection',
                    algorithm = 'StratifiedKFold',
                    default = {'n_splits': 5, 'shuffle': False},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    required = {'shuffle': True}),
                'time': Algorithm(
                    name = 'time',
                    module = 'sklearn.model_selection',
                    algorithm = 'TimeSeriesSplit',
                    default = {'n_splits': 5},
                    runtime = {'random_state': 'seed'},
                    selected = True),
                'train_test': Algorithm(
                    name = 'train_test',
                    module = 'sklearn.model_selection',
                    algorithm = 'ShuffleSplit',
                    default = {'test_size': 0.33},
                    runtime = {'random_state': 'seed'},
                    required = {'n_splits': 1},
                    selected = True)},
            'encoder': {
                'backward': Algorithm(
                    name = 'backward',
                    module = 'category_encoders',
                    algorithm = 'BackwardDifferenceEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'basen': Algorithm(
                    name = 'basen',
                    module = 'category_encoders',
                    algorithm = 'BaseNEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'binary': Algorithm(
                    name = 'binary',
                    module = 'category_encoders',
                    algorithm = 'BinaryEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'dummy': Algorithm(
                    name = 'dummy',
                    module = 'category_encoders',
                    algorithm = 'OneHotEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'hashing': Algorithm(
                    name = 'hashing',
                    module = 'category_encoders',
                    algorithm = 'HashingEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'helmert': Algorithm(
                    name = 'helmert',
                    module = 'category_encoders',
                    algorithm = 'HelmertEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'james_stein': Algorithm(
                    name = 'james_stein',
                    module = 'category_encoders',
                    algorithm = 'JamesSteinEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'loo': Algorithm(
                    name = 'loo',
                    module = 'category_encoders',
                    algorithm = 'LeaveOneOutEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'm_estimate': Algorithm(
                    name = 'm_estimate',
                    module = 'category_encoders',
                    algorithm = 'MEstimateEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'ordinal': Algorithm(
                    name = 'ordinal',
                    module = 'category_encoders',
                    algorithm = 'OrdinalEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'polynomial': Algorithm(
                    name = 'polynomial_encoder',
                    module = 'category_encoders',
                    algorithm = 'PolynomialEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'sum': Algorithm(
                    name = 'sum',
                    module = 'category_encoders',
                    algorithm = 'SumEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'target': Algorithm(
                    name = 'target',
                    module = 'category_encoders',
                    algorithm = 'TargetEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'woe': Algorithm(
                    name = 'weight_of_evidence',
                    module = 'category_encoders',
                    algorithm = 'WOEEncoder',
                    data_dependent = {'cols': 'categoricals'})},
            'mixer': {
                'polynomial': Algorithm(
                    name = 'polynomial_mixer',
                    module = 'sklearn.preprocessing',
                    algorithm = 'PolynomialFeatures',
                    default = {
                        'degree': 2,
                        'interaction_only': True,
                        'include_bias': True}),
                'quotient': Algorithm(
                    name = 'quotient',
                    module = None,
                    algorithm = 'QuotientFeatures'),
                'sum': Algorithm(
                    name = 'sum',
                    module = None,
                    algorithm = 'SumFeatures'),
                'difference': Algorithm(
                    name = 'difference',
                    module = None,
                    algorithm = 'DifferenceFeatures')},
            'cleaver': {},
            'sampler': {
                'adasyn': Algorithm(
                    name = 'adasyn',
                    module = 'imblearn.over_sampling',
                    algorithm = 'ADASYN',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'}),
                'cluster': Algorithm(
                    name = 'cluster',
                    module = 'imblearn.under_sampling',
                    algorithm = 'ClusterCentroids',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'}),
                'knn': Algorithm(
                    name = 'knn',
                    module = 'imblearn.under_sampling',
                    algorithm = 'AllKNN',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'}),
                'near_miss': Algorithm(
                    name = 'near_miss',
                    module = 'imblearn.under_sampling',
                    algorithm = 'NearMiss',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'}),
                'random_over': Algorithm(
                    name = 'random_over',
                    module = 'imblearn.over_sampling',
                    algorithm = 'RandomOverSampler',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'}),
                'random_under': Algorithm(
                    name = 'random_under',
                    module = 'imblearn.under_sampling',
                    algorithm = 'RandomUnderSampler',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'}),
                'smote': Algorithm(
                    name = 'smote',
                    module = 'imblearn.over_sampling',
                    algorithm = 'SMOTE',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'}),
                'smotenc': Algorithm(
                    name = 'smotenc',
                    module = 'imblearn.over_sampling',
                    algorithm = 'SMOTENC',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    data_dependent = {
                        'categorical_features': 'categoricals_indices'}),
                'smoteenn': Algorithm(
                    name = 'smoteenn',
                    module = 'imblearn.combine',
                    algorithm = 'SMOTEENN',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'}),
                'smotetomek': Algorithm(
                    name = 'smotetomek',
                    module = 'imblearn.combine',
                    algorithm = 'SMOTETomek',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'})},
            'reducer': {
                'kbest': Algorithm(
                    name = 'kbest',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectKBest',
                    default = {'k': 10, 'score_func': 'f_classif'},
                    selected = True),
                'fdr': Algorithm(
                    name = 'fdr',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectFdr',
                    default = {'alpha': 0.05, 'score_func': 'f_classif'},
                    selected = True),
                'fpr': Algorithm(
                    name = 'fpr',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectFpr',
                    default = {'alpha': 0.05, 'score_func': 'f_classif'},
                    selected = True),
                'custom': Algorithm(
                    name = 'custom',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectFromModel',
                    default = {'threshold': 'mean'},
                    runtime = {'estimator': 'algorithm'},
                    selected = True),
                'rank': Algorithm(
                    name = 'rank',
                    module = 'simplify.critic.rank',
                    algorithm = 'RankSelect',
                    selected = True),
                'rfe': Algorithm(
                    name = 'rfe',
                    module = 'sklearn.feature_selection',
                    algorithm = 'RFE',
                    default = {'n_features_to_select': 10, 'step': 1},
                    runtime = {'estimator': 'algorithm'},
                    selected = True),
                'rfecv': Algorithm(
                    name = 'rfecv',
                    module = 'sklearn.feature_selection',
                    algorithm = 'RFECV',
                    default = {'n_features_to_select': 10, 'step': 1},
                    runtime = {'estimator': 'algorithm'},
                    selected = True)}}
        model_options = {
            'classify': {
                'adaboost': Algorithm(
                    name = 'adaboost',
                    module = 'sklearn.ensemble',
                    algorithm = 'AdaBoostClassifier'),
                'baseline_classifier': Algorithm(
                    name = 'baseline_classifier',
                    module = 'sklearn.dummy',
                    algorithm = 'DummyClassifier',
                    required = {'strategy': 'most_frequent'}),
                'logit': Algorithm(
                    name = 'logit',
                    module = 'sklearn.linear_model',
                    algorithm = 'LogisticRegression'),
                'random_forest': Algorithm(
                    name = 'random_forest',
                    module = 'sklearn.ensemble',
                    algorithm = 'RandomForestClassifier'),
                'svm_linear': Algorithm(
                    name = 'svm_linear',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'linear', 'probability': True}),
                'svm_poly': Algorithm(
                    name = 'svm_poly',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'poly', 'probability': True}),
                'svm_rbf': Algorithm(
                    name = 'svm_rbf',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'rbf', 'probability': True}),
                'svm_sigmoid': Algorithm(
                    name = 'svm_sigmoid ',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'sigmoid', 'probability': True}),
                'tensorflow': Algorithm(
                    name = 'tensorflow',
                    module = 'tensorflow',
                    algorithm = None,
                    default = {
                        'batch_size': 10,
                        'epochs': 2}),
                'xgboost': Algorithm(
                    name = 'xgboost',
                    module = 'xgboost',
                    algorithm = 'XGBClassifier',
                    data_dependent = 'scale_pos_weight')},
            'cluster': {
                'affinity': Algorithm(
                    name = 'affinity',
                    module = 'sklearn.cluster',
                    algorithm = 'AffinityPropagation'),
                'agglomerative': Algorithm(
                    name = 'agglomerative',
                    module = 'sklearn.cluster',
                    algorithm = 'AgglomerativeClustering'),
                'birch': Algorithm(
                    name = 'birch',
                    module = 'sklearn.cluster',
                    algorithm = 'Birch'),
                'dbscan': Algorithm(
                    name = 'dbscan',
                    module = 'sklearn.cluster',
                    algorithm = 'DBSCAN'),
                'kmeans': Algorithm(
                    name = 'kmeans',
                    module = 'sklearn.cluster',
                    algorithm = 'KMeans'),
                'mean_shift': Algorithm(
                    name = 'mean_shift',
                    module = 'sklearn.cluster',
                    algorithm = 'MeanShift'),
                'spectral': Algorithm(
                    name = 'spectral',
                    module = 'sklearn.cluster',
                    algorithm = 'SpectralClustering'),
                'svm_linear': Algorithm(
                    name = 'svm_linear',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM'),
                'svm_poly': Algorithm(
                    name = 'svm_poly',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM'),
                'svm_rbf': Algorithm(
                    name = 'svm_rbf',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM,'),
                'svm_sigmoid': Algorithm(
                    name = 'svm_sigmoid',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM')},
            'regress': {
                'adaboost': Algorithm(
                    name = 'adaboost',
                    module = 'sklearn.ensemble',
                    algorithm = 'AdaBoostRegressor'),
                'baseline_regressor': Algorithm(
                    name = 'baseline_regressor',
                    module = 'sklearn.dummy',
                    algorithm = 'DummyRegressor',
                    required = {'strategy': 'mean'}),
                'bayes_ridge': Algorithm(
                    name = 'bayes_ridge',
                    module = 'sklearn.linear_model',
                    algorithm = 'BayesianRidge'),
                'lasso': Algorithm(
                    name = 'lasso',
                    module = 'sklearn.linear_model',
                    algorithm = 'Lasso'),
                'lasso_lars': Algorithm(
                    name = 'lasso_lars',
                    module = 'sklearn.linear_model',
                    algorithm = 'LassoLars'),
                'ols': Algorithm(
                    name = 'ols',
                    module = 'sklearn.linear_model',
                    algorithm = 'LinearRegression'),
                'random_forest': Algorithm(
                    name = 'random_forest',
                    module = 'sklearn.ensemble',
                    algorithm = 'RandomForestRegressor'),
                'ridge': Algorithm(
                    name = 'ridge',
                    module = 'sklearn.linear_model',
                    algorithm = 'Ridge'),
                'svm_linear': Algorithm(
                    name = 'svm_linear',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'linear', 'probability': True}),
                'svm_poly': Algorithm(
                    name = 'svm_poly',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'poly', 'probability': True}),
                'svm_rbf': Algorithm(
                    name = 'svm_rbf',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'rbf', 'probability': True}),
                'svm_sigmoid': Algorithm(
                    name = 'svm_sigmoid ',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'sigmoid', 'probability': True}),
                'xgboost': Algorithm(
                    name = 'xgboost',
                    module = 'xgboost',
                    algorithm = 'XGBRegressor',
                    data_dependent = 'scale_pos_weight')}}
        gpu_options = {
            'classify': {
                'forest_inference': Algorithm(
                    name = 'forest_inference',
                    module = 'cuml',
                    algorithm = 'ForestInference'),
                'random_forest': Algorithm(
                    name = 'random_forest',
                    module = 'cuml',
                    algorithm = 'RandomForestClassifier'),
                'logit': Algorithm(
                    name = 'logit',
                    module = 'cuml',
                    algorithm = 'LogisticRegression')},
            'cluster': {
                'dbscan': Algorithm(
                    name = 'dbscan',
                    module = 'cuml',
                    algorithm = 'DBScan'),
                'kmeans': Algorithm(
                    name = 'kmeans',
                    module = 'cuml',
                    algorithm = 'KMeans')},
            'regressor': {
                'lasso': Algorithm(
                    name = 'lasso',
                    module = 'cuml',
                    algorithm = 'Lasso'),
                'ols': Algorithm(
                    name = 'ols',
                    module = 'cuml',
                    algorithm = 'LinearRegression'),
                'ridge': Algorithm(
                    name = 'ridge',
                    module = 'cuml',
                    algorithm = 'RidgeRegression')}}
        self.dictionary['modeler'] = model_options[
            self.project.idea['chef']['model_type']]
        if self.project.idea['general']['gpu']:
            self.dictionary['modeler'].update(
                gpu_options[idea['chef']['model_type']])
        return self


@dataclass
class Cookbook(Book):
    """Stores recipes for preprocessing and machine learning.

    Args:
        project (Optional['Project']): related Project or subclass instance.
            Defaults to None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'chef'.
        steps (Optional[List[str], str]): ordered list of steps to execute. Each
            step should match a key in 'contents'. If a string is passed, it is
            converted to a 1-item list. Defaults to an empty list.
        contents (Optional['SimpleCatalog']): stores SimpleOutlines or
            subclasses in a SimpleCatalog instance which can be iterated in
            'chapters'. Defaults to an empty dictionary.
        chapters (Optional[List['Chapter']]): a list of Chapter instances that
            include a series of SimpleOutline or Technique instances to be applied
            to passed data. Defaults to an empty list.
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'recipes'.
        file_format (Optional[str]): file format to export the Book instance.
            Defaults to 'pickle'.
        export_folder (Optional[str]): the name of the attribute in the Project
            Inventory instance which corresponds to the export folder path to
            use when exporting a Book instance. Defaults to 'recipe'.
        returns_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    project: 'Project' = None
    name: Optional[str] = 'chef'
    steps: List[str] = field(default_factory = list)
    outline: Optional['SimpleOutline'] = None
    workers: 'SimpleCatalog' = field(default_factory = dict)
    chapters: List['Chapter'] = field(default_factory = list)
    iterable: Optional[str] = 'recipes'
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'recipe'
    returns_data: Optional[bool] = True

    """ Private Methods """

    def _calculate_hyperparameters(self) -> None:
        """Computes hyperparameters that can be determined by the source data
        (without creating data leakage problems).

        This method currently only support xgboost's scale_pos_weight
        parameter. Future hyperparameter computations will be added as they
        are discovered.

        """
        if self.steps['model'] in ['xgboost']:
            # Model class is injected with scale_pos_weight for algorithms that
            # use that parameter.
            self.model.scale_pos_weight = (
                    len(self.ingredients.y.index) /
                    ((self.ingredients.y == 1).sum())) - 1
        return self

    """ Public Tool Methods """

    def add_cleaves(self,
            cleave_group: str,
            prefixes: Union[List[str], str] = None,
            columns: Union[List[str], str] = None) -> None:
        """Adds cleaves to the list of cleaves.

        Args:
            cleave_group (str): names the set of features in the group.
            prefixes (Union[List[str], str]): name(s) of prefixes to columns to
                be included within the cleave.
            columns (Union[List[str], str]): name(s) of columns to be included
                within the cleave.

        """
        # if not self._exists('cleaves'):
        #     self.cleaves = []
        # columns = self.ingredients.make_column_list(
        #     prefixes = prefixes,
        #     columns = columns)
        # self.workers['cleaver'].add_techniques(
        #     cleave_group = cleave_group,
        #     columns = columns)
        # self.cleaves.append(cleave_group)
        return self
