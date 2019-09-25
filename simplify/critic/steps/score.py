
from dataclasses import dataclass

import pandas as pd
from sklearn import metrics

from simplify.core.base import SimpleStep


@dataclass
class Classifier(SimpleStep):

    def __post_init__(self):

        return self



    def _default_classifier(self):
        from sklearn import metrics
        self.options = {
                'accuracy' : metrics.accuracy_score,
                'balanced_accuracy' : metrics.balanced_accuracy_score,
                'f1' : metrics.f1_score,
                'f1_weighted' : metrics.f1_score,
                'fbeta' : metrics.fbeta_score,
                'hamming' : metrics.hamming_loss,
                'jaccard' : metrics.jaccard_similarity_score,
                'matthews_corrcoef' : metrics.matthews_corrcoef,
                'neg_log_loss' :  metrics.log_loss,
                'precision' :  metrics.precision_score,
                'precision_weighted' :  metrics.precision_score,
                'recall' :  metrics.recall_score,
                'recall_weighted' :  metrics.recall_score,
                'zero_one' : metrics.zero_one_loss}
        self.prob_options = {'brier_score_loss' : metrics.brier_score_loss}
        self.score_options = {'roc_auc' :  metrics.roc_auc_score}
        return self

    def _default_clusterer(self):
        from sklearn import metrics
        self.options = {
                'adjusted_mutual_info' : metrics.adjusted_mutual_info_score,
                'adjusted_rand' : metrics.adjusted_rand_score,
                'calinski' : metrics.calinski_harabasz_score,
                'davies' : metrics.davies_bouldin_score,
                'completeness' : metrics.completeness_score,
                'contingency_matrix' : metrics.cluster.contingency_matrix,
                'fowlkes' : metrics.fowlkes_mallows_score,
                'h_completness' : metrics.homogeneity_completeness_v_measure,
                'homogeniety' : metrics.homogeneity_score,
                'mutual_info' : metrics.mutual_info_score,
                'norm_mutual_info' : metrics.normalized_mutual_info_score,
                'silhouette' : metrics.silhouette_score,
                'v_measure' : metrics.v_measure_score}
        self.prob_options = {}
        self.score_options = {}
        return self

    def _default_regressor(self):
        from sklearn import metrics
        self.options = {
                'explained_variance' : metrics.explained_variance_score,
                'max_error' : metrics.max_error,
                'absolute_error' : metrics.absolute_error,
                'mse' : metrics.mean_squared_error,
                'msle' : metrics.mean_squared_log_error,
                'mae' : metrics.median_absolute_error,
                'r2' : metrics.r2_score}
        self.prob_options = {}
        self.score_options = {}
        return self



