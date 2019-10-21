
"""
.. module:: techniques
:synopsis: default techniques for siMpLify package
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections import namedtuple

Technique = namedtuple('name',
    ['module', 'algorithm', 'default_parameters', 'extra_parameters',
     'runtime_parameters', 'selected_parameters', 'conditional_parameters'])

""" Scale Techniques """

scale_bins = Technique(
    name = 'bins',
    module = 'sklearn.preprocessing',
    algorithm = 'KBinsDiscretizer',
    default_parameters = {
        'encode': 'ordinal', 'strategy': 'uniform', 'n_bins': 5})
scale_gauss = Technique(
    name = 'gauss',
    module = 'simplify.chef.steps.techniques.gaussify',
    algorithm = 'Gaussify',
    default_parameters = {'standardize': False, 'copy': False})
scale_maxabs = Technique(
    name = 'maxabs',
    module = 'sklearn.preprocessing',
    algorithm = 'MaxAbsScaler',
    default_parameters = {'copy': False})
scale_minmax = Technique(
    name = 'minmax',
    module = 'sklearn.preprocessing',
    algorithm = 'MinMaxScaler',
    default_parameters = {'copy': False})
scale_normalize = Technique(
    name = 'normalize',
    module = 'sklearn.preprocessing',
    algorithm = 'Normalizer',
    default_parameters = {'copy': False})
scale_quantile = Technique(
    name = 'quantile',
    module = 'sklearn.preprocessing',
    algorithm = 'QuantileTransformer',
    default_parameters = {'copy': False})
scale_robust = Technique(
    name = 'robust',
    module = 'sklearn.preprocessing',
    algorithm = 'RobustScaler',
    default_parameters = {'copy': False})
scale_standard = Technique(
    name = 'standard',
    module = 'sklearn.preprocessing',
    algorithm = 'StandardScaler',
    default_parameters = {'copy': False})

""" Split Techniques """

split_group_kfold = Technique(
    name = 'group_kfold',
    module = 'sklearn.model_selection',
    algorithm = 'GroupKFold',
    default_parameters = {'n_splits': 5},
    runtime_parameters = {'random_state': 'seed'},
    selected_parameters = True)
split_kfold = Technique(
    name = 'kfold',
    module = 'sklearn.model_selection',
    algorithm = 'KFold',
    default_parameters = {'n_splits': 5, 'shuffle': False},
    runtime_parameters = {'random_state': 'seed'},
    selected_parameters = True)
split_stratified = Technique(
    name = 'stratified',
    module = 'sklearn.model_selection',
    algorithm = 'StratifiedKFold',
    default_parameters = {'n_splits': 5, 'shuffle': False},
    runtime_parameters = {'random_state': 'seed'},
    selected_parameters = True)
split_time = Technique(
    name = 'time',
    module = 'sklearn.model_selection',
    algorithm = 'TimeSeriesSplit',
    default_parameters = {'n_splits': 5},
    runtime_parameters = {'random_state': 'seed'},
    selected_parameters = True)
split_train_test = Technique(
    name = 'train_test',
    module = 'sklearn.model_selection',
    algorithm = 'ShuffleSplit',
    default_parameters = {'test_size': 0.33},
    runtime_parameters = {'random_state': 'seed'},
    extra_parameters = {'n_splits': 1},
    selected_parameters = True)

""" Encode Techniques """

encode_backward = Technique(
    name = 'backward',
    module = 'category_encoders',
    algorithm = 'BackwardDifferenceEncoder',
    runtime_parameters = {'cols': 'columns'})
encode_basen = Technique(
    name = 'basen',
    module = 'category_encoders',
    algorithm = 'BaseNEncoder',
    runtime_parameters = {'cols': 'columns'})
encode_binary = Technique(
    name = 'binary',
    module = 'category_encoders',
    algorithm = 'BinaryEncoder',
    runtime_parameters = {'cols': 'columns'})
encode_dummy = Technique(
    name = 'dummy',
    module = 'category_encoders',
    algorithm = 'OneHotEncoder',
    runtime_parameters = {'cols': 'columns'})
encode_hashing = Technique(
    name = 'hashing',
    module = 'category_encoders',
    algorithm = 'HashingEncoder',
    runtime_parameters = {'cols': 'columns'})
encode_helmert = Technique(
    name = 'helmert',
    module = 'category_encoders',
    algorithm = 'HelmertEncoder',
    runtime_parameters = {'cols': 'columns'})
encode_loo = Technique(
    name = 'loo',
    module = 'category_encoders',
    algorithm = 'LeaveOneOutEncoder',
    runtime_parameters = {'cols': 'columns'})
encode_ordinal = Technique(
    name = 'ordinal',
    module = 'category_encoders',
    algorithm = 'OrdinalEncoder',
    runtime_parameters = {'cols': 'columns'})
encode_sum = Technique(
    name = 'sum',
    module = 'category_encoders',
    algorithm = 'SumEncoder',
    runtime_parameters = {'cols': 'columns'})
encode_target = Technique(
    name = 'target',
    module = 'category_encoders',
    algorithm = 'TargetEncoder',
    runtime_parameters = {'cols': 'columns'})

""" Mix Techniques """

mix_polynomial = Technique(
    name = 'polynomial',
    module = 'category_encoders',
    algorithm = 'PolynomialEncoder',
    runtime_parameters = {'cols': 'columns'})
mix_quotient = Technique(
    name = 'quotient',
    module = 'simplify.chef.steps.techniques.mixers',
    algorithm = 'QuotientFeatures',
    runtime_parameters = {'cols': 'columns'})
mix_sum = Technique(
    name = 'sum',
    module = 'simplify.chef.steps.techniques.mixers',
    algorithm = 'SumFeatures',
    runtime_parameters = {'cols': 'columns'})
mix_difference = Technique(
    name = 'difference',
    module = 'simplify.chef.steps.techniques.mixers',
    algorithm = 'DifferenceFeatures',
    runtime_parameters = {'cols': 'columns'})

""" Cleave Techniques """

cleave_compare = Technique(
    name = 'compare',
    module = 'simplify.chef.steps.techniques.cleavers',
    algorithm = 'CompareCleaves')
cleave_combine = Technique(
    name = 'combine',
    module = 'simplify.chef.steps.techniques.cleavers',
    algorithm = 'CombineCleaves')

""" Sample Techniques """

sample_adasyn = Technique(
    name = 'adasyn',
    module = 'imblearn.over_sampling',
    algorithm = 'ADASYN',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'})
sample_cluster = Technique(
    name = 'cluster',
    module = 'imblearn.under_sampling',
    algorithm = 'ClusterCentroids',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'})
sample_knn = Technique(
    name = 'knn',
    module = 'imblearn.under_sampling',
    algorithm = 'AllKNN',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'})
sample_near_miss = Technique(
    name = 'near_miss',
    module = 'imblearn.under_sampling',
    algorithm = 'NearMiss',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'})
sample_random_over = Technique(
    name = 'random_over',
    module = 'imblearn.over_sampling',
    algorithm = 'RandomOverSampler',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'})
sample_random_under = Technique(
    name = 'random_under',
    module = 'imblearn.under_sampling',
    algorithm = 'RandomUnderSampler',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'})
sample_smote = Technique(
    name = 'smote',
    module = 'imblearn.over_sampling',
    algorithm = 'SMOTE',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'})
sample_smotenc = Technique(
    name = 'smotenc',
    module = 'imblearn.over_sampling',
    algorithm = 'SMOTENC',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'},
    conditional_parameters = True)
sample_smoteenn = Technique(
    name = 'smoteenn',
    module = 'imblearn.combine',
    algorithm = 'SMOTEENN',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'})
sample_smotetomek = Technique(
    name = 'smotetomek',
    module = 'imblearn.combine',
    algorithm = 'SMOTETomek',
    default_parameters = {'sampling_strategy': 'auto'},
    runtime_parameters = {'random_state': 'seed'})

def _recheck_parameters(parameters, ingredients, columns = None):
    if self.technique in ['smotenc']:
        if columns:
            cat_features = self._get_indices(ingredients.x, columns)
            self.parameters.update({'categorical_features': cat_features})
        else:
            cat_features = self._get_indices(ingredients.x,
                                                ingredients.categoricals)
    return parameters

""" Reduce Techniques """

reduce_kbest = Technique(
    name = 'kbest',
    module = 'sklearn.feature_selection',
    algorithm = 'SelectKBest',
    default_parameters = {'k': 10, 'score_func': f_classif})
reduce_fdr = Technique(
    name = 'fdr',
    module = 'sklearn.feature_selection',
    algorithm = 'SelectFdr',
    default_parameters = {'alpha': 0.05, 'score_func': f_classif})
reduce_fpr = Technique(
    name = 'fpr',
    module = 'sklearn.feature_selection',
    algorithm = 'SelectFpr',
    default_parameters = {'alpha': 0.05, 'score_func': f_classif})
reduce_custom = Technique(
    name = 'custom',
    module = 'sklearn.feature_selection',
    algorithm = 'SelectFromModel',
    default_parameters = {'threshold': 'mean'},
    runtime_parameters = {'estimator': 'estimator'})
reduce_rank = Technique(
    name = 'rank',
    module = 'simplify.critic.rank',
    algorithm = 'RankSelect')
reduce_rfe = Technique(
    name = 'rfe',
    module = 'sklearn.feature_selection',
    algorithm = 'RFE',
    default_parameters = {'n_features_to_select': 10, 'step': 1},
    runtime_parameters = {'estimator': 'estimator'})
reduce_rfecv = Technique(
    name = 'rfecv',
    module = 'sklearn.feature_selection',
    algorithm = 'RFECV',
    default_parameters = {'n_features_to_select': 10, 'step': 1},
    runtime_parameters = {'estimator': 'estimator'})

""" Model Techniques """

classify_adaboost = Technique(
    name = 'adaboost',
    module = 'sklearn.ensemble',
    algorithm = 'AdaBoostClassifier')
classify_baseline_classifier = Technique(
    name = 'baseline_classifier',
    module = 'sklearn.dummy',
    algorithm = 'DummyClassifier',
    extra_parameters = {'strategy': 'most_frequent'})
classify_logit = Technique(
    name = 'logit',
    module = 'sklearn.linear_model',
    algorithm = 'LogisticRegression')
classify_random_forest = Technique(
    name = 'random_forest',
    module = 'sklearn.ensemble',
    algorithm = 'RandomForestClassifier')
classify_svm_linear = Technique(
    name = 'svm_linear',
    module = 'sklearn.svm',
    algorithm = 'SVC',
    extra_parameters = {'kernel': 'linear', 'probability': True})
classify_svm_poly = Technique(
    name = 'svm_poly',
    module = 'sklearn.svm',
    algorithm = 'SVC',
    extra_parameters = {'kernel': 'poly', 'probability': True})
classify_svm_rbf = Technique(
    name = 'svm_rbf',
    module = 'sklearn.svm',
    algorithm = 'SVC',
    extra_parameters = {'kernel': 'rbf', 'probability': True})
classify_svm_sigmoid = Technique(
    name = 'svm_sigmoid ',
    module = 'sklearn.svm',
    algorithm = 'SVC',
    extra_parameters = {'kernel': 'sigmoid', 'probability': True})
classify_xgboost = Technique(
    name = 'xgboost',
    module = 'xgboost',
    algorithm = 'XGBClassifier')
gpu_classify_forest_inference = Technique(
    name = 'forest_inference',
    module = 'cuml',
    algorithm = 'ForestInference')
gpu_classify_random_forest = Technique(
    name = 'random_forest',
    module = 'cuml',
    algorithm = 'RandomForestClassifier')
gpu_classify_logit = Technique(
    name = 'logit',
    module = 'cuml',
    algorithm = 'LogisticRegression')

cluster_affinity = Technique(
    name = 'affinity',
    module = 'sklearn.cluster',
    algorithm = 'AffinityPropagation')
cluster_agglomerative = Technique(
    name = 'agglomerative',
    module = 'sklearn.cluster',
    algorithm = 'AgglomerativeClustering')
cluster_birch = Technique(
    name = 'birch',
    module = 'sklearn.cluster',
    algorithm = 'Birch')
cluster_dbscan = Technique(
    name = 'dbscan',
    module = 'sklearn.cluster',
    algorithm = 'DBSCAN')
cluster_kmeans = Technique(
    name = 'kmeans',
    module = 'sklearn.cluster',
    algorithm = 'KMeans')
cluster_mean_shift = Technique(
    name = 'mean_shift',
    module = 'kbest',
    module = 'sklearn.cluster',
    algorithm = 'MeanShift')
cluster_spectral = Technique(
    name = 'spectral',
    module = 'sklearn.cluster',
    algorithm = 'SpectralClustering')
cluster_svm_linear = Technique(
    name = 'svm_linear',
    module = 'sklearn.cluster',
    algorithm = 'OneClassSVM')
cluster_svm_poly = Technique(
    name = 'svm_poly',
    module = 'sklearn.cluster',
    algorithm = 'OneClassSVM')
cluster_svm_rbf = Technique(
    name = 'svm_rbf',
    module = 'sklearn.cluster',
    algorithm = 'OneClassSVM,')
cluster_svm_sigmoid = Technique(
    name = 'svm_sigmoid',
    module = 'sklearn.cluster',
    algorithm = 'OneClassSVM')
gpu_cluster_dbscan = Technique(
    name = 'dbscan',
    module = 'cuml',
    algorithm = 'DBScan')
gpu_cluster_kmeans = Technique(
    name = 'kmeans',
    module = 'cuml',
    algorithm = 'KMeans')

regress_adaboost = Technique(
    name = 'adaboost',
    module = 'sklearn.ensemble',
    algorithm = 'AdaBoostRegressor')
regress_baseline_regressor = Technique(
    name = 'baseline_regressor',
    module = 'sklearn.dummy',
    algorithm = 'DummyRegressor',
    extra_parameters = {'strategy': 'mean'})
regress_bayes_ridge = Technique(
    name = 'bayes_ridge',
    module = 'sklearn.linear_model',
    algorithm = 'BayesianRidge')
regress_lasso = Technique(
    name = 'lasso',
    module = 'sklearn.linear_model',
    algorithm = 'Lasso')
regress_lasso_lars = Technique(
    name = 'lasso_lars',
    module = 'sklearn.linear_model',
    algorithm = 'LassoLars')
regress_ols = Technique(
    name = 'ols',
    module = 'sklearn.linear_model',
    algorithm = 'LinearRegression')
regress_random_forest = Technique(
    name = 'random_forest',
    module = 'sklearn.ensemble',
    algorithm = 'RandomForestRegressor')
regress_ridge = Technique(
    name = 'ridge',
    module = 'sklearn.linear_model',
    algorithm = 'Ridge')
regress_svm_linear = Technique(
    name = 'svm_linear',
    module = 'sklearn.svm',
    algorithm = 'SVC',
    extra_parameters = {'kernel': 'linear', 'probability': True})
regress_svm_poly = Technique(
    name = 'svm_poly',
    module = 'sklearn.svm',
    algorithm = 'SVC',
    extra_parameters = {'kernel': 'poly', 'probability': True})
regress_svm_rbf = Technique(
    name = 'svm_rbf',
    module = 'sklearn.svm',
    algorithm = 'SVC',
    extra_parameters = {'kernel': 'rbf', 'probability': True})
regress_svm_sigmoid = Technique(
    name = 'svm_sigmoid ',
    module = 'sklearn.svm',
    algorithm = 'SVC',
    extra_parameters = {'kernel': 'sigmoid', 'probability': True})
regress_xgboost = Technique(
    name = 'xgboost',
    module = 'xgboost',
    algorithm = 'XGBRegressor')
gpu_regress_lasso = Technique(
    name = 'lasso',
    module = 'cuml',
    algorithm = 'Lasso')
gpu_regress_ols = Technique(
    name = 'ols',
    module = 'cuml',
    algorithm = 'LinearRegression')
gpu_regress_ridge = Technique(
    name = 'ridge',
    module = 'cuml',
    algorithm = 'RidgeRegression')

""" Search Techniques """

search_bayes = Technique(
    name = 'bayes',
     module = 'skopt',
    algorithm = 'BayesSearchCV')
search_grid = Technique(
    name = 'grid',
    module = 'sklearn.model_selection',
    algorithm = 'GridSearchCV',
    runtime_parameters = {
        'estimator': 'estimator.algorithm',
        'param_distributions': 'space',
        'random_state': 'seed'},
    conditional_parameters = True)
search_random = Technique(
    name = 'random',
    module = 'sklearn.model_selection',
    algorithm = 'RandomizedSearchCV',
    runtime_parameters = {
        'estimator': 'estimator.algorithm',
        'param_distributions': 'space',
        'random_state': 'seed'},
    conditional_parameters = True)

def conditional_search_parameters(parameters):
    if 'refit' in parameters and isinstance(parameters['scoring'], list):
        parameters['scoring'] = parameters['scoring'][0]
    return parameters