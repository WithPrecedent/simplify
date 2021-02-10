"""
criteria:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import simplify
import sourdough 


criteria: sourdough.types.Catalog[str, simplify.SimpleTechnique] = (
    sourdough.types.Catalog(contents = {
        'accuracy': simplify.SimpleTechnique(
            name = 'accuracy',
            module = 'sklearn.metrics',
            algorithm = 'accuracy_score'),
        'adjusted_mutual_info': simplify.SimpleTechnique(
            name = 'adjusted_mutual_info_score',
            module = 'sklearn.metrics',
            algorithm = 'adjusted_mutual_info'),
        'adjusted_rand': simplify.SimpleTechnique(
            name = 'adjusted_rand',
            module = 'sklearn.metrics',
            algorithm = 'adjusted_rand_score'),
        'balanced_accuracy': simplify.SimpleTechnique(
            name = 'balanced_accuracy',
            module = 'sklearn.metrics',
            algorithm = 'balanced_accuracy_score'),
        'brier_score_loss': simplify.SimpleTechnique(
            name = 'brier_score_loss',
            module = 'sklearn.metrics',
            algorithm = 'brier_score_loss'),
        'calinski': simplify.SimpleTechnique(
            name = 'calinski_harabasz',
            module = 'sklearn.metrics',
            algorithm = 'calinski_harabasz_score'),
        'davies': simplify.SimpleTechnique(
            name = 'davies_bouldin',
            module = 'sklearn.metrics',
            algorithm = 'davies_bouldin_score'),
        'completeness': simplify.SimpleTechnique(
            name = 'completeness',
            module = 'sklearn.metrics',
            algorithm = 'completeness_score'),
        'contingency_matrix': simplify.SimpleTechnique(
            name = 'contingency_matrix',
            module = 'sklearn.metrics',
            algorithm = 'cluster.contingency_matrix'),
        'explained_variance': simplify.SimpleTechnique(
            name = 'explained_variance',
            module = 'sklearn.metrics',
            algorithm = 'explained_variance_score'),
        'f1': simplify.SimpleTechnique(
            name = 'f1',
            module = 'sklearn.metrics',
            algorithm = 'f1_score'),
        'f1_weighted': simplify.SimpleTechnique(
            name = 'f1_weighted',
            module = 'sklearn.metrics',
            algorithm = 'f1_score',
            required = {'average': 'weighted'}),
        'fbeta': simplify.SimpleTechnique(
            name = 'fbeta',
            module = 'sklearn.metrics',
            algorithm = 'fbeta_score',
            required = {'beta': 1}),
        'fowlkes': simplify.SimpleTechnique(
            name = 'fowlkes_mallows',
            module = 'sklearn.metrics',
            algorithm = 'fowlkes_mallows_score'),
        'hamming': simplify.SimpleTechnique(
            name = 'hamming_loss',
            module = 'sklearn.metrics',
            algorithm = 'hamming_loss'),
        'h_completness': simplify.SimpleTechnique(
            name = 'homogeneity_completeness',
            module = 'sklearn.metrics',
            algorithm = 'homogeneity_completeness_v_measure'),
        'homogeniety': simplify.SimpleTechnique(
            name = 'homogeneity',
            module = 'sklearn.metrics',
            algorithm = 'homogeneity_score'),
        'jaccard': simplify.SimpleTechnique(
            name = 'jaccard_similarity',
            module = 'sklearn.metrics',
            algorithm = 'jaccard_similarity_score'),
        'mae': simplify.SimpleTechnique(
            name = 'median_absolute_error',
            module = 'sklearn.metrics',
            algorithm = 'median_absolute_error'),
        'matthews_corrcoef': simplify.SimpleTechnique(
            name = 'matthews_correlation_coefficient',
            module = 'sklearn.metrics',
            algorithm = 'matthews_corrcoef'),
        'max_error': simplify.SimpleTechnique(
            name = 'max_error',
            module = 'sklearn.metrics',
            algorithm = 'max_error'),
        'mean_absolute_error': simplify.SimpleTechnique(
            name = 'mean_absolute_error',
            module = 'sklearn.metrics',
            algorithm = 'mean_absolute_error'),
        'mse': simplify.SimpleTechnique(
            name = 'mean_squared_error',
            module = 'sklearn.metrics',
            algorithm = 'mean_squared_error'),
        'msle': simplify.SimpleTechnique(
            name = 'mean_squared_log_error',
            module = 'sklearn.metrics',
            algorithm = 'mean_squared_log_error'),
        'mutual_info': simplify.SimpleTechnique(
            name = 'mutual_info_score',
            module = 'sklearn.metrics',
            algorithm = 'mutual_info_score'),
        'log_loss': simplify.SimpleTechnique(
            name = 'log_loss',
            module = 'sklearn.metrics',
            algorithm = 'log_loss'),
        'norm_mutual_info': simplify.SimpleTechnique(
            name = 'normalized_mutual_info',
            module = 'sklearn.metrics',
            algorithm = 'normalized_mutual_info_score'),
        'precision': simplify.SimpleTechnique(
            name = 'precision',
            module = 'sklearn.metrics',
            algorithm = 'precision_score'),
        'precision_weighted': simplify.SimpleTechnique(
            name = 'precision_weighted',
            module = 'sklearn.metrics',
            algorithm = 'precision_score',
            required = {'average': 'weighted'}),
        'r2': simplify.SimpleTechnique(
            name = 'r2',
            module = 'sklearn.metrics',
            algorithm = 'r2_score'),
        'recall': simplify.SimpleTechnique(
            name = 'recall',
            module = 'sklearn.metrics',
            algorithm = 'recall_score'),
        'recall_weighted': simplify.SimpleTechnique(
            name = 'recall_weighted',
            module = 'sklearn.metrics',
            algorithm = 'recall_score',
            required = {'average': 'weighted'}),
        'roc_auc': simplify.SimpleTechnique(
            name = 'roc_auc',
            module = 'sklearn.metrics',
            algorithm = 'roc_auc_score'),
        'silhouette': simplify.SimpleTechnique(
            name = 'silhouette',
            module = 'sklearn.metrics',
            algorithm = 'silhouette_score'),
        'v_measure': simplify.SimpleTechnique(
            name = 'v_measure',
            module = 'sklearn.metrics',
            algorithm = 'v_measure_score'),
        'zero_one': simplify.SimpleTechnique(
            name = 'zero_one',
            module = 'sklearn.metrics',
            algorithm = 'zero_one_loss')}))
