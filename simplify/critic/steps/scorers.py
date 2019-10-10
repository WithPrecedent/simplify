"""
.. module:: scorers
:synopsis: metrics and reports for model performance
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd
from sklearn import metrics

from simplify.core.technique import SimpleTechnique


@dataclass
class Metrics(SimpleTechnique):

    technique: object = None
    parameters: object = None
    name: str = 'metrics'

    def __post_init__(self):
        self.idea_sections = ['critic']
        super().__post_init__()
        return self

    def draft(self):
        self.options = {
            'accuracy': metrics.accuracy_score,
            'adjusted_mutual_info': metrics.adjusted_mutual_info_score,
            'adjusted_rand': metrics.adjusted_rand_score,
            'balanced_accuracy': metrics.balanced_accuracy_score,
            'brier_score_loss': metrics.brier_score_loss,
            'calinski': metrics.calinski_harabasz_score,
            'davies': metrics.davies_bouldin_score,
            'completeness': metrics.completeness_score,
            'contingency_matrix': metrics.cluster.contingency_matrix,
            'explained_variance': metrics.explained_variance_score,
            'f1': metrics.f1_score,
            'f1_weighted': metrics.f1_score,
            'fbeta': metrics.fbeta_score,
            'fowlkes': metrics.fowlkes_mallows_score,
            'hamming': metrics.hamming_loss,
            'h_completness': metrics.homogeneity_completeness_v_measure,
            'homogeniety': metrics.homogeneity_score,
            'jaccard': metrics.jaccard_similarity_score,
            'mae': metrics.median_absolute_error,
            'matthews_corrcoef': metrics.matthews_corrcoef,
            'max_error': metrics.max_error,
            'mean_absolute_error': metrics.mean_absolute_error,
            'mse': metrics.mean_squared_error,
            'msle': metrics.mean_squared_log_error,
            'mutual_info': metrics.mutual_info_score,
            'neg_log_loss':  metrics.log_loss,
            'norm_mutual_info': metrics.normalized_mutual_info_score,
            'precision':  metrics.precision_score,
            'precision_weighted':  metrics.precision_score,
            'r2': metrics.r2_score,
            'recall':  metrics.recall_score,
            'recall_weighted':  metrics.recall_score,
            'roc_auc':  metrics.roc_auc_score,
            'silhouette': metrics.silhouette_score,
            'v_measure': metrics.v_measure_score,
            'zero_one': metrics.zero_one_loss}
        self.prob_options = ['brier_score_loss']
        self.score_options = ['roc_auc']
        self.negative_options = ['brier_loss_score', 'neg_log_loss',
                                 'zero_one']
        self.special_options = {
            'fbeta': {'beta': 1},
            'f1_weighted': {'average': 'weighted'},
            'precision_weighted': {'average': 'weighted'},
            'recall_weighted': {'average': 'weighted'}}
        return self

@dataclass
class Tests(SimpleTechnique):

    technique: object = None
    parameters: object = None
    name: str = 'tests'

    def __post_init__(self):
        self.idea_sections = ['critic']
        super().__post_init__()
        return self

    def draft(self):
        self.options = {
            'ks_distribution': ['scipy.stats', 'ks_2samp'],
            'ks_goodness': ['scipy.stats', 'kstest'],
            'kurtosis_test': ['scipy.stats', 'kurtosistest'],
            'normal': ['scipy.stats', 'normaltest'],
            'pearson': ['scipy.stats.pearsonr']}
        return self

@dataclass
class Reports(SimpleTechnique):

    technique: object = None
    parameters: object = None
    name: str = 'reports'

    def __post_init__(self):
        self.idea_sections = ['critic']
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def _classifier_report(self):
        self.classifier_report_default = metrics.classification_report(
                self.recipe.ingredients.y_test,
                self.predictions)
        self.classifier_report_dict = metrics.classification_report(
                self.recipe.ingredients.y_test,
                self.predictions,
                output_dict = True)
        self.classifier_report = pd.DataFrame(
                self.classifier_report_dict).transpose()
        return self

    def _confusion_matrix(self):
        self.confusion = metrics.confusion_matrix(
                self.recipe.ingredients.y_test, self.predictions)
        return self

    def _cluster_report(self):
        return self

    def _regressor_report(self):
        return self

    def _print_classifier_results(self, recipe):
        """Prints to console basic results separate from report."""
        print('These are the results using the', recipe.model.technique,
              'model')
        if recipe.splicer.technique != 'none':
            print('Testing', recipe.splicer.technique, 'predictors')
        print('Confusion Matrix:')
        print(self.confusion)
        print('Classification Report:')
        print(self.classification_report)
        return self
