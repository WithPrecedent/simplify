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
from simplify.core.book import TechniqueOutline


OPTIONS = {
    'filler': {
        'defaults': TechniqueOutline(
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
        'impute': TechniqueOutline(
            name = 'defaults',
            module = 'sklearn.impute',
            component = 'SimpleImputer',
            default = {'defaults': {}}),
        'knn_impute': TechniqueOutline(
            name = 'defaults',
            module = 'sklearn.impute',
            component = 'KNNImputer',
            default = {'defaults': {}})},
    'categorizer': {
        'automatic': TechniqueOutline(
            name = 'automatic',
            module = 'simplify.chef.algorithms',
            component = 'auto_categorize',
            default = {'threshold': 10}),
        'binary': TechniqueOutline(
            name = 'binary',
            module = 'sklearn.preprocessing',
            component = 'Binarizer',
            default = {'threshold': 0.5}),
        'bins': TechniqueOutline(
            name = 'bins',
            module = 'sklearn.preprocessing',
            component = 'KBinsDiscretizer',
            default = {
                'strategy': 'uniform',
                'n_bins': 5},
            selected = True,
            required = {'encode': 'onehot'})},
    'scaler': {
        'gauss': TechniqueOutline(
            name = 'gauss',
            module = None,
            component = 'Gaussify',
            default = {'standardize': False, 'copy': False},
            selected = True,
            required = {'rescaler': 'standard'}),
        'maxabs': TechniqueOutline(
            name = 'maxabs',
            module = 'sklearn.preprocessing',
            component = 'MaxAbsScaler',
            default = {'copy': False},
            selected = True),
        'minmax': TechniqueOutline(
            name = 'minmax',
            module = 'sklearn.preprocessing',
            component = 'MinMaxScaler',
            default = {'copy': False},
            selected = True),
        'normalize': TechniqueOutline(
            name = 'normalize',
            module = 'sklearn.preprocessing',
            component = 'Normalizer',
            default = {'copy': False},
            selected = True),
        'quantile': TechniqueOutline(
            name = 'quantile',
            module = 'sklearn.preprocessing',
            component = 'QuantileTransformer',
            default = {'copy': False},
            selected = True),
        'robust': TechniqueOutline(
            name = 'robust',
            module = 'sklearn.preprocessing',
            component = 'RobustScaler',
            default = {'copy': False},
            selected = True),
        'standard': TechniqueOutline(
            name = 'standard',
            module = 'sklearn.preprocessing',
            component = 'StandardScaler',
            default = {'copy': False},
            selected = True)},
    'splitter': {
        'group_kfold': TechniqueOutline(
            name = 'group_kfold',
            module = 'sklearn.model_selection',
            component = 'GroupKFold',
            default = {'n_splits': 5},
            runtime = {'random_state': 'seed'},
            selected = True),
        'kfold': TechniqueOutline(
            name = 'kfold',
            module = 'sklearn.model_selection',
            component = 'KFold',
            default = {'n_splits': 5, 'shuffle': False},
            runtime = {'random_state': 'seed'},
            selected = True,
            required = {'shuffle': True}),
        'stratified': TechniqueOutline(
            name = 'stratified',
            module = 'sklearn.model_selection',
            component = 'StratifiedKFold',
            default = {'n_splits': 5, 'shuffle': False},
            runtime = {'random_state': 'seed'},
            selected = True,
            required = {'shuffle': True}),
        'time': TechniqueOutline(
            name = 'time',
            module = 'sklearn.model_selection',
            component = 'TimeSeriesSplit',
            default = {'n_splits': 5},
            runtime = {'random_state': 'seed'},
            selected = True),
        'train_test': TechniqueOutline(
            name = 'train_test',
            module = 'sklearn.model_selection',
            component = 'ShuffleSplit',
            default = {'test_size': 0.33},
            runtime = {'random_state': 'seed'},
            required = {'n_splits': 1},
            selected = True)},
    'encoder': {
        'backward': TechniqueOutline(
            name = 'backward',
            module = 'category_encoders',
            component = 'BackwardDifferenceEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'basen': TechniqueOutline(
            name = 'basen',
            module = 'category_encoders',
            component = 'BaseNEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'binary': TechniqueOutline(
            name = 'binary',
            module = 'category_encoders',
            component = 'BinaryEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'dummy': TechniqueOutline(
            name = 'dummy',
            module = 'category_encoders',
            component = 'OneHotEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'hashing': TechniqueOutline(
            name = 'hashing',
            module = 'category_encoders',
            component = 'HashingEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'helmert': TechniqueOutline(
            name = 'helmert',
            module = 'category_encoders',
            component = 'HelmertEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'james_stein': TechniqueOutline(
            name = 'james_stein',
            module = 'category_encoders',
            component = 'JamesSteinEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'loo': TechniqueOutline(
            name = 'loo',
            module = 'category_encoders',
            component = 'LeaveOneOutEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'm_estimate': TechniqueOutline(
            name = 'm_estimate',
            module = 'category_encoders',
            component = 'MEstimateEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'ordinal': TechniqueOutline(
            name = 'ordinal',
            module = 'category_encoders',
            component = 'OrdinalEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'polynomial': TechniqueOutline(
            name = 'polynomial_encoder',
            module = 'category_encoders',
            component = 'PolynomialEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'sum': TechniqueOutline(
            name = 'sum',
            module = 'category_encoders',
            component = 'SumEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'target': TechniqueOutline(
            name = 'target',
            module = 'category_encoders',
            component = 'TargetEncoder',
            data_dependent = {'cols': 'categoricals'}),
        'woe': TechniqueOutline(
            name = 'weight_of_evidence',
            module = 'category_encoders',
            component = 'WOEEncoder',
            data_dependent = {'cols': 'categoricals'})},
    'mixer': {
        'polynomial': TechniqueOutline(
            name = 'polynomial_mixer',
            module = 'sklearn.preprocessing',
            component = 'PolynomialFeatures',
            default = {
                'degree': 2,
                'interaction_only': True,
                'include_bias': True}),
        'quotient': TechniqueOutline(
            name = 'quotient',
            module = None,
            component = 'QuotientFeatures'),
        'sum': TechniqueOutline(
            name = 'sum',
            module = None,
            component = 'SumFeatures'),
        'difference': TechniqueOutline(
            name = 'difference',
            module = None,
            component = 'DifferenceFeatures')},
    'cleaver': {},
    'sampler': {
        'adasyn': TechniqueOutline(
            name = 'adasyn',
            module = 'imblearn.over_sampling',
            component = 'ADASYN',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'cluster': TechniqueOutline(
            name = 'cluster',
            module = 'imblearn.under_sampling',
            component = 'ClusterCentroids',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'knn': TechniqueOutline(
            name = 'knn',
            module = 'imblearn.under_sampling',
            component = 'AllKNN',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'near_miss': TechniqueOutline(
            name = 'near_miss',
            module = 'imblearn.under_sampling',
            component = 'NearMiss',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'random_over': TechniqueOutline(
            name = 'random_over',
            module = 'imblearn.over_sampling',
            component = 'RandomOverSampler',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'random_under': TechniqueOutline(
            name = 'random_under',
            module = 'imblearn.under_sampling',
            component = 'RandomUnderSampler',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'smote': TechniqueOutline(
            name = 'smote',
            module = 'imblearn.over_sampling',
            component = 'SMOTE',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'smotenc': TechniqueOutline(
            name = 'smotenc',
            module = 'imblearn.over_sampling',
            component = 'SMOTENC',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'},
            data_dependent = {'categorical_features': 'categoricals_indices'}),
        'smoteenn': TechniqueOutline(
            name = 'smoteenn',
            module = 'imblearn.combine',
            component = 'SMOTEENN',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'}),
        'smotetomek': TechniqueOutline(
            name = 'smotetomek',
            module = 'imblearn.combine',
            component = 'SMOTETomek',
            default = {'sampling_strategy': 'auto'},
            runtime = {'random_state': 'seed'})},
    'reducer': {
        'kbest': TechniqueOutline(
            name = 'kbest',
            module = 'sklearn.feature_selection',
            component = 'SelectKBest',
            default = {'k': 10, 'score_func': 'f_classif'},
            selected = True),
        'fdr': TechniqueOutline(
            name = 'fdr',
            module = 'sklearn.feature_selection',
            component = 'SelectFdr',
            default = {'alpha': 0.05, 'score_func': 'f_classif'},
            selected = True),
        'fpr': TechniqueOutline(
            name = 'fpr',
            module = 'sklearn.feature_selection',
            component = 'SelectFpr',
            default = {'alpha': 0.05, 'score_func': 'f_classif'},
            selected = True),
        'custom': TechniqueOutline(
            name = 'custom',
            module = 'sklearn.feature_selection',
            component = 'SelectFromModel',
            default = {'threshold': 'mean'},
            runtime = {'estimator': 'algorithm'},
            selected = True),
        'rank': TechniqueOutline(
            name = 'rank',
            module = 'simplify.critic.rank',
            component = 'RankSelect',
            selected = True),
        'rfe': TechniqueOutline(
            name = 'rfe',
            module = 'sklearn.feature_selection',
            component = 'RFE',
            default = {'n_features_to_select': 10, 'step': 1},
            runtime = {'estimator': 'algorithm'},
            selected = True),
        'rfecv': TechniqueOutline(
            name = 'rfecv',
            module = 'sklearn.feature_selection',
            component = 'RFECV',
            default = {'n_features_to_select': 10, 'step': 1},
            runtime = {'estimator': 'algorithm'},
            selected = True)}}

MODEL_OPTIONS = {
    'classify': {
        'adaboost': TechniqueOutline(
            name = 'adaboost',
            module = 'sklearn.ensemble',
            component = 'AdaBoostClassifier'),
        'baseline_classifier': TechniqueOutline(
            name = 'baseline_classifier',
            module = 'sklearn.dummy',
            component = 'DummyClassifier',
            required = {'strategy': 'most_frequent'}),
        'logit': TechniqueOutline(
            name = 'logit',
            module = 'sklearn.linear_model',
            component = 'LogisticRegression'),
        'random_forest': TechniqueOutline(
            name = 'random_forest',
            module = 'sklearn.ensemble',
            component = 'RandomForestClassifier'),
        'svm_linear': TechniqueOutline(
            name = 'svm_linear',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'linear', 'probability': True}),
        'svm_poly': TechniqueOutline(
            name = 'svm_poly',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'poly', 'probability': True}),
        'svm_rbf': TechniqueOutline(
            name = 'svm_rbf',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'rbf', 'probability': True}),
        'svm_sigmoid': TechniqueOutline(
            name = 'svm_sigmoid ',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'sigmoid', 'probability': True}),
        'tensorflow': TechniqueOutline(
            name = 'tensorflow',
            module = 'tensorflow',
            component = None,
            default = {
                'batch_size': 10,
                'epochs': 2}),
        'xgboost': TechniqueOutline(
            name = 'xgboost',
            module = 'xgboost',
            component = 'XGBClassifier',
            data_dependent = 'scale_pos_weight')},
    'cluster': {
        'affinity': TechniqueOutline(
            name = 'affinity',
            module = 'sklearn.cluster',
            component = 'AffinityPropagation'),
        'agglomerative': TechniqueOutline(
            name = 'agglomerative',
            module = 'sklearn.cluster',
            component = 'AgglomerativeClustering'),
        'birch': TechniqueOutline(
            name = 'birch',
            module = 'sklearn.cluster',
            component = 'Birch'),
        'dbscan': TechniqueOutline(
            name = 'dbscan',
            module = 'sklearn.cluster',
            component = 'DBSCAN'),
        'kmeans': TechniqueOutline(
            name = 'kmeans',
            module = 'sklearn.cluster',
            component = 'KMeans'),
        'mean_shift': TechniqueOutline(
            name = 'mean_shift',
            module = 'sklearn.cluster',
            component = 'MeanShift'),
        'spectral': TechniqueOutline(
            name = 'spectral',
            module = 'sklearn.cluster',
            component = 'SpectralClustering'),
        'svm_linear': TechniqueOutline(
            name = 'svm_linear',
            module = 'sklearn.cluster',
            component = 'OneClassSVM'),
        'svm_poly': TechniqueOutline(
            name = 'svm_poly',
            module = 'sklearn.cluster',
            component = 'OneClassSVM'),
        'svm_rbf': TechniqueOutline(
            name = 'svm_rbf',
            module = 'sklearn.cluster',
            component = 'OneClassSVM,'),
        'svm_sigmoid': TechniqueOutline(
            name = 'svm_sigmoid',
            module = 'sklearn.cluster',
            component = 'OneClassSVM')},
    'regress': {
        'adaboost': TechniqueOutline(
            name = 'adaboost',
            module = 'sklearn.ensemble',
            component = 'AdaBoostRegressor'),
        'baseline_regressor': TechniqueOutline(
            name = 'baseline_regressor',
            module = 'sklearn.dummy',
            component = 'DummyRegressor',
            required = {'strategy': 'mean'}),
        'bayes_ridge': TechniqueOutline(
            name = 'bayes_ridge',
            module = 'sklearn.linear_model',
            component = 'BayesianRidge'),
        'lasso': TechniqueOutline(
            name = 'lasso',
            module = 'sklearn.linear_model',
            component = 'Lasso'),
        'lasso_lars': TechniqueOutline(
            name = 'lasso_lars',
            module = 'sklearn.linear_model',
            component = 'LassoLars'),
        'ols': TechniqueOutline(
            name = 'ols',
            module = 'sklearn.linear_model',
            component = 'LinearRegression'),
        'random_forest': TechniqueOutline(
            name = 'random_forest',
            module = 'sklearn.ensemble',
            component = 'RandomForestRegressor'),
        'ridge': TechniqueOutline(
            name = 'ridge',
            module = 'sklearn.linear_model',
            component = 'Ridge'),
        'svm_linear': TechniqueOutline(
            name = 'svm_linear',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'linear', 'probability': True}),
        'svm_poly': TechniqueOutline(
            name = 'svm_poly',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'poly', 'probability': True}),
        'svm_rbf': TechniqueOutline(
            name = 'svm_rbf',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'rbf', 'probability': True}),
        'svm_sigmoid': TechniqueOutline(
            name = 'svm_sigmoid ',
            module = 'sklearn.svm',
            component = 'SVC',
            required = {'kernel': 'sigmoid', 'probability': True}),
        'xgboost': TechniqueOutline(
            name = 'xgboost',
            module = 'xgboost',
            component = 'XGBRegressor',
            data_dependent = 'scale_pos_weight')}}

GPU_OPTIONS = {
    'classify': {
        'forest_inference': TechniqueOutline(
            name = 'forest_inference',
            module = 'cuml',
            component = 'ForestInference'),
        'random_forest': TechniqueOutline(
            name = 'random_forest',
            module = 'cuml',
            component = 'RandomForestClassifier'),
        'logit': TechniqueOutline(
            name = 'logit',
            module = 'cuml',
            component = 'LogisticRegression')},
    'cluster': {
        'dbscan': TechniqueOutline(
            name = 'dbscan',
            module = 'cuml',
            component = 'DBScan'),
        'kmeans': TechniqueOutline(
            name = 'kmeans',
            module = 'cuml',
            component = 'KMeans')},
    'regressor': {
        'lasso': TechniqueOutline(
            name = 'lasso',
            module = 'cuml',
            component = 'Lasso'),
        'ols': TechniqueOutline(
            name = 'ols',
            module = 'cuml',
            component = 'LinearRegression'),
        'ridge': TechniqueOutline(
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
    options['modeler'] = MODEL_OPTIONS[idea['chef']['model_type']]
    if idea['general']['gpu']:
        options['modeler'].update(GPU_OPTIONS[idea['chef']['model_type']])
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
        contents (Optional['Contents']): stores SimpleOutlines or
            subclasses in a Contents instance which can be iterated in
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
    library: 'Contents' = field(default_factory = dict)
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
        # self.library['cleaver'].add_techniques(
        #     cleave_group = cleave_group,
        #     columns = columns)
        # self.cleaves.append(cleave_group)
        return self
