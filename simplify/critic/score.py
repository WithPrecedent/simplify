"""
.. module:: score
:synopsis: records metrics for model performance
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd
from sklearn import metrics

from simplify.core.base import SimplePlan, SimpleStep


@dataclass
class Score(SimplePlan):
    """Scores models and prepares reports based upon model type.
    
    Args:
        steps(dict(str: SimpleStep)): names and related SimpleStep classes for
            explaining data analysis models.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_produce (bool): whether to call the 'produce' method when the class
            is instanced.
    """
    steps: object = None
    name: str = 'scorer'
    auto_publish: bool = True
    auto_produce: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _set_columns(self):
        self.columns = list(self.options.keys())
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
                'metrics': Metrics,
                'reports': Reports}

        self.checks = ['steps']
        return self

    def edit(self, name, metric, special_type = None,
             special_parameters = None, negative_metric = False):
        """Allows user to manually add a metric to report."""
        self.options.update({name: metric})
        if special_type in ['probability']:
            self.prob_options.update({name: metric})
        elif special_type in ['scorer']:
            self.score_options.update({name: metric})
        if special_parameters:
           self.special_options.update({name: special_parameters})
        if negative_metric:
           self.negative_options.append[name]
        return self

    def publish(self):
        self._set_columns()
        return self

    def produce(self, recipe):
        """Prepares the results of a single recipe application to be added to
        the .report dataframe.
        """
        scores = pd.Series(index = self.columns)
        for column, value in self.options.items():
            if column in self.metrics:
                if column in self.prob_options:
                    params = {'y_true': self.recipe.ingredients.y_test,
                              'y_prob': self.predicted_probs[:, 1]}
                elif column in self.score_options:
                    params = {'y_true': self.recipe.ingredients.y_test,
                              'y_score': self.predicted_probs[:, 1]}
                else:
                    params = {'y_true': self.recipe.ingredients.y_test,
                              'y_pred': self.predictions}
                if column in self.special_metrics:
                    params.update({column: self.special_metrics[column]})
                result = value(**params)
                if column in self.negative_metrics:
                    result = -1 * result
                scores[column] = result
        return scores

@dataclass
class Metrics(SimpleStep):


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
class Reports(SimpleStep):

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
