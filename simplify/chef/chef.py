"""
.. module:: chef
:synopsis: machine learning made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.core.book import Book
from simplify.core.book import PageOutline


OPTIONS = {
    'filler': {
        'defaults': PageOutline(
            name = 'defaults',
            module = 'simplify.chef.algorithms',
            component = 'smart_fill',
            default = {'defaults': {
                'boolean': False,
                'float': 0.0,
                'integer': 0,
                'string': '',
                'categorical': '',
                'list': [],
                'datetime': 1/1/1900,
                'timedelta': 0}}),
        'impute': PageOutline(
            name = 'defaults',
            module = 'sklearn.impute',
            component = 'SimpleImputer',
            default = {'defaults': {}}),
        'knn_impute': PageOutline(
            name = 'defaults',
            module = 'sklearn.impute',
            component = 'KNNImputer',
            default = {'defaults': {}})},
    'categorizer': {
        'automatic': PageOutline(
            name = 'automatic',
            module = 'simplify.chef.algorithms',
            component = 'auto_categorize',
            default = {'threshold': 10}),
        'binary': PageOutline(
            name = 'binary',
            module = 'sklearn.preprocessing',
            component = 'Binarizer',
            default = {'threshold': 0.5}),
        'bins': PageOutline(
            name = 'bins',
            module = 'sklearn.preprocessing',
            component = 'KBinsDiscretizer',
            default = {
                'strategy': 'uniform',
                'n_bins': 5},
            selected = True,
            required = {'encode': 'onehot'})},
    'scaler': {
        'gauss': PageOutline(
            name = 'gauss',
            module = None,
            component = 'Gaussify',
            default = {'standardize': False, 'copy': False},
            selected = True,
            required = {'rescaler': 'standard'}),
        'maxabs': PageOutline(
            name = 'maxabs',
            module = 'sklearn.preprocessing',
            component = 'MaxAbsScaler',
            default = {'copy': False},
            selected = True),
        'minmax': PageOutline(
            name = 'minmax',
            module = 'sklearn.preprocessing',
            component = 'MinMaxScaler',
            default = {'copy': False},
            selected = True),
        'normalize': PageOutline(
            name = 'normalize',
            module = 'sklearn.preprocessing',
            component = 'Normalizer',
            default = {'copy': False},
            selected = True),
        'quantile': PageOutline(
            name = 'quantile',
            module = 'sklearn.preprocessing',
            component = 'QuantileTransformer',
            default = {'copy': False},
            selected = True),
        'robust': PageOutline(
            name = 'robust',
            module = 'sklearn.preprocessing',
            component = 'RobustScaler',
            default = {'copy': False},
            selected = True),
        'standard': PageOutline(
            name = 'standard',
            module = 'sklearn.preprocessing',
            component = 'StandardScaler',
            default = {'copy': False},
            selected = True)},
    'splitter': {
        'group_kfold': PageOutline(
            name = 'group_kfold',
            module = 'sklearn.model_selection',
            component = 'GroupKFold',
            default = {'n_splits': 5},
            runtime = {'random_state': 'seed'},
            selected = True),
        'kfold': PageOutline(
            name = 'kfold',
            module = 'sklearn.model_selection',
            component = 'KFold',
            default = {'n_splits': 5, 'shuffle': False},
            runtime = {'random_state': 'seed'},
            selected = True,
            required = {'shuffle': True}),
        'stratified': PageOutline(
            name = 'stratified',
            module = 'sklearn.model_selection',
            component = 'StratifiedKFold',
            default = {'n_splits': 5, 'shuffle': False},
            runtime = {'random_state': 'seed'},
            selected = True,
            required = {'shuffle': True}),
        'time': PageOutline(
            name = 'time',
            module = 'sklearn.model_selection',
            component = 'TimeSeriesSplit',
            default = {'n_splits': 5},
            runtime = {'random_state': 'seed'},
            selected = True),
        'train_test': PageOutline(
            name = 'train_test',
            module = 'sklearn.model_selection',
            component = 'ShuffleSplit',
            default = {'test_size': 0.33},
            runtime = {'random_state': 'seed'},
            required = {'n_splits': 1},
            selected = True)},
    'encoder': {
        'backward': PageOutline(
            name = 'backward',
            module = 'category_encoders',
            component = 'BackwardDifferenceEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'basen': PageOutline(
            name = 'basen',
            module = 'category_encoders',
            component = 'BaseNEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'binary': PageOutline(
            name = 'binary',
            module = 'category_encoders',
            component = 'BinaryEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'dummy': PageOutline(
            name = 'dummy',
            module = 'category_encoders',
            component = 'OneHotEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'hashing': PageOutline(
            name = 'hashing',
            module = 'category_encoders',
            component = 'HashingEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'helmert': PageOutline(
            name = 'helmert',
            module = 'category_encoders',
            component = 'HelmertEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'james_stein': PageOutline(
            name = 'james_stein',
            module = 'category_encoders',
            component = 'JamesSteinEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'loo': PageOutline(
            name = 'loo',
            module = 'category_encoders',
            component = 'LeaveOneOutEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'm_estimate': PageOutline(
            name = 'm_estimate',
            module = 'category_encoders',
            component = 'MEstimateEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'ordinal': PageOutline(
            name = 'ordinal',
            module = 'category_encoders',
            component = 'OrdinalEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'polynomial': PageOutline(
            name = 'polynomial_encoder',
            module = 'category_encoders',
            component = 'PolynomialEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'sum': PageOutline(
            name = 'sum',
            module = 'category_encoders',
            component = 'SumEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'target': PageOutline(
            name = 'target',
            module = 'category_encoders',
            component = 'TargetEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'woe': PageOutline(
            name = 'weight_of_evidence',
            module = 'category_encoders',
            component = 'WOEEncoder',
            data_dependent = {'cols': 'categoricals'})},
    'mixer': {
        'polynomial': PageOutline(
            name = 'polynomial_mixer',
            module = 'sklearn.preprocessing',
            component = 'PolynomialFeatures',
            default = {
                'degree': 2,
                'interaction_only': True,
                'include_bias': True}),
        'quotient': PageOutline(
            name = 'quotient',
            module = None,
            component = 'QuotientFeatures'),
        'sum': PageOutline(
            name = 'sum',
            module = None,
            component = 'SumFeatures'),
        'difference': PageOutline(
            name = 'difference',
            module = None,
            component = 'DifferenceFeatures')},
    'cleaver': {},
    'sampler': {
        'adasyn': PageOutline(
            name = 'adasyn',
            module = 'imblearn.over_sampling',
            component = 'ADASYN',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'cluster': PageOutline(
            name = 'cluster',
            module = 'imblearn.under_sampling',
            component = 'ClusterCentroids',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'knn': PageOutline(
            name = 'knn',
            module = 'imblearn.under_sampling',
            component = 'AllKNN',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'near_miss': PageOutline(
            name = 'near_miss',
            module = 'imblearn.under_sampling',
            component = 'NearMiss',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'random_over': PageOutline(
            name = 'random_over',
            module = 'imblearn.over_sampling',
            component = 'RandomOverSampler',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'random_under': PageOutline(
            name = 'random_under',
            module = 'imblearn.under_sampling',
            component = 'RandomUnderSampler',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'smote': PageOutline(
            name = 'smote',
            module = 'imblearn.over_sampling',
            component = 'SMOTE',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'smotenc': PageOutline(
            name = 'smotenc',
            module = 'imblearn.over_sampling',
            component = 'SMOTENC',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            data_dependent = {'categorical_features': 'categoricals_indices'}),
        'smoteenn': PageOutline(
            name = 'smoteenn',
            module = 'imblearn.combine',
            component = 'SMOTEENN',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'smotetomek': PageOutline(
            name = 'smotetomek',
            module = 'imblearn.combine',
            component = 'SMOTETomek',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'})},
    'reducer': {
        'kbest': PageOutline(
            name = 'kbest',
            module = 'sklearn.feature_selection',
            component = 'SelectKBest',
            default = {'k': 10, 'score_func': 'f_classif'},
            selected = True),
        'fdr': PageOutline(
            name = 'fdr',
            module = 'sklearn.feature_selection',
            component = 'SelectFdr',
            default = {'alpha': 0.05, 'score_func': 'f_classif'},
            selected = True),
        'fpr': PageOutline(
            name = 'fpr',
            module = 'sklearn.feature_selection',
            component = 'SelectFpr',
            default = {'alpha': 0.05, 'score_func': 'f_classif'},
            selected = True),
        'custom': PageOutline(
            name = 'custom',
            module = 'sklearn.feature_selection',
            component = 'SelectFromModel',
            default = {'threshold': 'mean'},
            runtime = {'estimator': 'algorithm'},
            selected = True),
        'rank': PageOutline(
            name = 'rank',
            module = 'simplify.critic.rank',
            component = 'RankSelect',
            selected = True),
        'rfe': PageOutline(
            name = 'rfe',
            module = 'sklearn.feature_selection',
            component = 'RFE',
            default = {'n_features_to_select': 10, 'step': 1},
            runtime = {'estimator': 'algorithm'},
            selected = True),
        'rfecv': PageOutline(
            name = 'rfecv',
            module = 'sklearn.feature_selection',
            component = 'RFECV',
            default = {'n_features_to_select': 10, 'step': 1},
            runtime = {'estimator': 'algorithm'},
            selected = True)}}

MODEL_OPTIONS = {
    'classifier': {
        'adaboost': PageOutline(
            name = 'adaboost',
            module = 'sklearn.ensemble',
            component = 'AdaBoostClassifier'),
        'baseline_classifier': PageOutline(
            name = 'baseline_classifier',
            module = 'sklearn.dummy',
            component = 'DummyClassifier',
            required = {'strategy': 'most_frequent'}),
        'logit': PageOutline(
            name = 'logit',
            module = 'sklearn.linear_model',
            component = 'LogisticRegression'),
        'random_forest': PageOutline(
            name = 'random_forest',
            module = 'sklearn.ensemble',
            component = 'RandomForestClassifier'),
        'svm_linear': PageOutline(
            name = 'svm_linear',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'linear', 'probability': True}),
        'svm_poly': PageOutline(
            name = 'svm_poly',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'poly', 'probability': True}),
        'svm_rbf': PageOutline(
            name = 'svm_rbf',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'rbf', 'probability': True}),
        'svm_sigmoid': PageOutline(
            name = 'svm_sigmoid ',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'sigmoid', 'probability': True}),
        'tensorflow': PageOutline(
            name = 'tensorflow',
            module = 'tensorflow',
            component = None,
            default = {
                'batch_size': 10,
                'epochs': 2}),
        'xgboost': PageOutline(
            name = 'xgboost',
            module = 'xgboost',
            component = 'XGBClassifier',
            data_dependent = 'scale_pos_weight')},
    'cluster': {
        'affinity': PageOutline(
            name = 'affinity',
            module = 'sklearn.cluster',
            component = 'AffinityPropagation'),
        'agglomerative': PageOutline(
            name = 'agglomerative',
            module = 'sklearn.cluster',
            component = 'AgglomerativeClustering'),
        'birch': PageOutline(
            name = 'birch',
            module = 'sklearn.cluster',
            component = 'Birch'),
        'dbscan': PageOutline(
            name = 'dbscan',
            module = 'sklearn.cluster',
            component = 'DBSCAN'),
        'kmeans': PageOutline(
            name = 'kmeans',
            module = 'sklearn.cluster',
            component = 'KMeans'),
        'mean_shift': PageOutline(
            name = 'mean_shift',
            module = 'sklearn.cluster',
            component = 'MeanShift'),
        'spectral': PageOutline(
            name = 'spectral',
            module = 'sklearn.cluster',
            component = 'SpectralClustering'),
        'svm_linear': PageOutline(
            name = 'svm_linear',
            module = 'sklearn.cluster',
            component = 'OneClassSVM'),
        'svm_poly': PageOutline(
            name = 'svm_poly',
            module = 'sklearn.cluster',
            component = 'OneClassSVM'),
        'svm_rbf': PageOutline(
            name = 'svm_rbf',
            module = 'sklearn.cluster',
            component = 'OneClassSVM,'),
        'svm_sigmoid': PageOutline(
            name = 'svm_sigmoid',
            module = 'sklearn.cluster',
            component = 'OneClassSVM')},
    'regressor': {
        'adaboost': PageOutline(
            name = 'adaboost',
            module = 'sklearn.ensemble',
            component = 'AdaBoostRegressor'),
        'baseline_regressor': PageOutline(
            name = 'baseline_regressor',
            module = 'sklearn.dummy',
            component = 'DummyRegressor',
            required = {'strategy': 'mean'}),
        'bayes_ridge': PageOutline(
            name = 'bayes_ridge',
            module = 'sklearn.linear_model',
            component = 'BayesianRidge'),
        'lasso': PageOutline(
            name = 'lasso',
            module = 'sklearn.linear_model',
            component = 'Lasso'),
        'lasso_lars': PageOutline(
            name = 'lasso_lars',
            module = 'sklearn.linear_model',
            component = 'LassoLars'),
        'ols': PageOutline(
            name = 'ols',
            module = 'sklearn.linear_model',
            component = 'LinearRegression'),
        'random_forest': PageOutline(
            name = 'random_forest',
            module = 'sklearn.ensemble',
            component = 'RandomForestRegressor'),
        'ridge': PageOutline(
            name = 'ridge',
            module = 'sklearn.linear_model',
            component = 'Ridge'),
        'svm_linear': PageOutline(
            name = 'svm_linear',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'linear', 'probability': True}),
        'svm_poly': PageOutline(
            name = 'svm_poly',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'poly', 'probability': True}),
        'svm_rbf': PageOutline(
            name = 'svm_rbf',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'rbf', 'probability': True}),
        'svm_sigmoid': PageOutline(
            name = 'svm_sigmoid ',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'sigmoid', 'probability': True}),
        'xgboost': PageOutline(
            name = 'xgboost',
            module = 'xgboost',
            component = 'XGBRegressor',
            data_dependent = 'scale_pos_weight')}}

GPU_OPTIONS = {
    'classify': {
        'forest_inference': PageOutline(
            name = 'forest_inference',
            module = 'cuml',
            component = 'ForestInference'),
        'random_forest': PageOutline(
            name = 'random_forest',
            module = 'cuml',
            component = 'RandomForestClassifier'),
        'logit': PageOutline(
            name = 'logit',
            module = 'cuml',
            component = 'LogisticRegression')},
    'cluster': {
        'dbscan': PageOutline(
            name = 'dbscan',
            module = 'cuml',
            component = 'DBScan'),
        'kmeans': PageOutline(
            name = 'kmeans',
            module = 'cuml',
            component = 'KMeans')},
    'regressor': {
        'lasso': PageOutline(
            name = 'lasso',
            module = 'cuml',
            component = 'Lasso'),
        'ols': PageOutline(
            name = 'ols',
            module = 'cuml',
            component = 'LinearRegression'),
        'ridge': PageOutline(
            name = 'ridge',
            module = 'cuml',
            component = 'RidgeRegression')}}

def get_options(idea: 'Idea') -> Dict:
    """Returns options for chef package.

    Args:
        idea ('Idea'): an Idea instance.

    Returns:
        Dict: with options based upon settings in 'idea'.

    """
    options = OPTIONS
    options['model'] = MODEL_OPTIONS[idea['chef']['model_type']]
    if idea['general']['gpu']:
        options['model'].update(GPU_OPTIONS[idea['chef']['model_type']])
    return options


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
        contents (Optional['SimpleContents']): stores SimpleOutlines or
            subclasses in a SimpleContents instance which can be iterated in
            'chapters'. Defaults to an empty dictionary.
        chapters (Optional[List['Chapter']]): a list of Chapter instances that
            include a series of SimpleOutline or Page instances to be applied
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
    library: 'SimpleContents' = field(default_factory = dict)
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
        # self.library['cleaver'].add_pages(
        #     cleave_group = cleave_group,
        #     columns = columns)
        # self.cleaves.append(cleave_group)
        return self
