"""
.. module:: defaults
:synopsis: stored default settings for siMpLify
:publisher: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""


GENERAL = {
    'verbose': True,
    'seed': 43,
    'conserve_memory': True,
    'parallelize': False,
    'gpu': False}

PROJECT = {
    'project_workers': ['analyze', 'criticize']}

FILES = {
    'source_format': 'csv',
    'interim_format': 'csv',
    'final_format': 'csv',
    'analysis_format': 'csv',
    'file_encoding': 'windows-1252',
    'float_format': '%.4f',
    'test_data': True,
    'test_chunk': 500,
    'random_test_chunk': True,
    'boolean_out': True,
    'naming_classes': ['model', 'cleave'],
    'export_results': True}

FARMER = {
    'wrangler_steps': None}

CHEF = {
    'analyst_steps': ['scale', 'split', 'encode', 'sample', 'reduce', 'model'],
    'data_to_use': 'train_test',
    'model_type': 'classify',
    'label': 'target',
    'calculate_hyperparameters': False,
    'export_all_recipes': True,
    'fill_techniques': [None],
    'categorize_techniques': [None],
    'scale_techniques': ['normalize', 'minmax'],
    'split_techniques': ['train_test'],
    'encode_techniques': ['target'],
    'mix_techniques': ['polynomial'],
    'cleave_techniques': [None],
    'sample_techniques': ['smote', 'adasyn'],
    'reduce_techniques': [None],
    'model_techniques': ['xgboost'],
    'search_step': 'random'}

ACTUARY = {
    'explorer_steps': 'summary',
    'summary_techniques': ['default'],
    'test_techniques': None}

CRITIC = {
    'critic_steps': [
        'predict', 'probability', 'explain', 'rank', 'measure', 'report'],
    'predict_techniques': 'gini',
    'probability_techniques': ['gini', 'log'],
    'explain_techniques': ['eli5', 'shap'],
    'ranker_techniques': ['gini', 'shap'],
    'measure_techniques': ['all'],
    'report_techniques': ['default'],
    'data_to_review': 'test',
    'join_predictions': True,
    'join_probabilities': True}

ARTIST = {
    'artist_steps': ['illustrate', 'paint'],
    'illustrator_techniques': ['default'],
    'painter_techniques': ['default'],
    'animator_techniques': ['default'],
    'data_to_plot': 'test',
    'comparison_plots': False}

STYLER_PARAMETERS = {
    'plot_style': 'fivethirtyeight',
    'plot_font': 'Franklin Gothic Book',
    'seaborn_style': 'darkgrid',
    'seaborn_context': 'paper',
    'seaborn_palette': 'dark',
    'interactions_display': 10,
    'features_display': 20,
    'summary_display': 20,
    'dependency_plots': ['cleave', 'top_features'],
    'shap_plot_type': 'dot'}

SCALER_PARAMETERS = {
    'copy': False,
    'encode': 'ordinal',
    'strategy': 'uniform',
    'n_bins': 5}

SPLITTER_PARAMETERS = {
    'test_size': 0.33,
    'val_size': 0,
    'n_splits': 5,
    'shuffle': False}

ENCODER_PARAMETERS = {}

MIXER_PARAMETERS = {}

CLEAVER_PARAMETERS = {
    'include_all': True}

SAMPLER_PARAMETERS = {
    'sampling_strategy': 'auto'}

REDUCER_PARAMETERS = {
    'n_features_to_select': 10,
    'step': 1,
    'score_func': 'f_classif',
    'alpha': 0.05,
    'threshold': 'mean'}

SEARCH_PARAMETERS = {
    'n_iter': 50,
    'scoring': ['roc_auc', 'f1', 'neg_log_loss'],
    'cv': 5,
    'refit': True}

RANDOM_FOREST_PARAMETERS = {
    'n_estimators': [20, 1000],
    'max_depth': [5, 30],
    'max_features': [10, 50],
    'max_leaf_nodes': [5, 10],
    'bootstrap': True,
    'oob_score': True,
    'verbose': 0}

XGBOOST_PARAMETERS = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'silent': True,
    'n_estimators': 50,
    'max_depth': 5,
    'learning_rate': 0.001,
    'subsample': 0.3,
    'colsample_bytree': 0.3,
    'colsample_bylevel': 0.3,
    'min_child_weight': 0.7,
    'gamma': 0.0,
    'alpha': 0.0}

TENSORFLOW_PARAMETERS = {}

BASELINE_CLASSIFIER_PARAMETERS = {
    'strategy': 'most_frequent'}
