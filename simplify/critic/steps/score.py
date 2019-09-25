
from dataclasses import dataclass

import pandas as pd
from sklearn import metrics

from simplify.core.base import SimpleStep, SimpleTechnique


@dataclass
class Score(SimpleStep):
    """Core class for evaluating the results of data analysis produceed by
    the siMpLify Cookbook.

    """
    steps : object = None
    name : str = 'scorer'
    auto_finalize : bool = True
    auto_produce : bool = False

    def __post_init__(self):
        """Sets up the core attributes of an Evaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        self.options = {
            'accuracy' : metrics.accuracy_score,
            'adjusted_mutual_info' : metrics.adjusted_mutual_info_score,
            'adjusted_rand' : metrics.adjusted_rand_score,
            'balanced_accuracy' : metrics.balanced_accuracy_score,
            'brier_score_loss' : metrics.brier_score_loss,
            'calinski' : metrics.calinski_harabasz_score,
            'davies' : metrics.davies_bouldin_score,
            'completeness' : metrics.completeness_score,
            'contingency_matrix' : metrics.cluster.contingency_matrix,
            'explained_variance' : metrics.explained_variance_score,
            'f1' : metrics.f1_score,
            'f1_weighted' : metrics.f1_score,
            'fbeta' : metrics.fbeta_score,
            'fowlkes' : metrics.fowlkes_mallows_score,
            'hamming' : metrics.hamming_loss,
            'h_completness' : metrics.homogeneity_completeness_v_measure,
            'homogeniety' : metrics.homogeneity_score,
            'jaccard' : metrics.jaccard_similarity_score,
            'mae' : metrics.median_absolute_error,
            'matthews_corrcoef' : metrics.matthews_corrcoef,
            'max_error' : metrics.max_error,
            'mean_absolute_error' : metrics.mean_absolute_error,
            'mse' : metrics.mean_squared_error,
            'msle' : metrics.mean_squared_log_error,
            'mutual_info' : metrics.mutual_info_score,
            'neg_log_loss' :  metrics.log_loss,
            'norm_mutual_info' : metrics.normalized_mutual_info_score,
            'precision' :  metrics.precision_score,
            'precision_weighted' :  metrics.precision_score,
            'r2' : metrics.r2_score,
            'recall' :  metrics.recall_score,
            'recall_weighted' :  metrics.recall_score,
            'roc_auc' :  metrics.roc_auc_score,
            'silhouette' : metrics.silhouette_score,
            'v_measure' : metrics.v_measure_score,
            'zero_one' : metrics.zero_one_loss}
        self.prob_options = ['brier_score_loss']
        self.score_options = ['roc_auc']
        self.negative_options = ['brier_loss_score', 'neg_log_loss',
                                 'zero_one']
        self.special_options = {
            'fbeta' : {'beta' : 1},
            'f1_weighted' : {'average' : 'weighted'},
            'precision_weighted' : {'average' : 'weighted'},
            'recall_weighted' : {'average' : 'weighted'}}
        self.checks = ['idea']
        return self

    def edit(self, name, metric, special_type = None,
             special_parameters = None, negative_metric = False):
        """Allows user to manually add a metric to report."""
        self.options.update({name : metric})
        if special_type in ['probability']:
            self.prob_options.update({name : metric})
        elif special_type in ['scorer']:
            self.score_options.update({name : metric})
        if special_parameters:
           self.special_options.update({name : special_parameters})
        if negative_metric:
           self.negative_options.append[name]
        return self

    def produce(self):
        """Prepares the results of a single recipe application to be added to
        the .report dataframe.
        """
        self.result = pd.Series(index = self.columns_list)
        for column, value in self.columns.items():
            if isinstance(getattr(self.recipe, value), CookbookSimpleStep):
                self.result[column] = self._format_step(value)
            else:
                self.result[column] = getattr(self.recipe, value)
        for column, value in self.options.items():
            if column in self.metrics:
                if column in self.prob_options:
                    params = {'y_true' : self.recipe.ingredients.y_test,
                              'y_prob' : self.predicted_probs[:, 1]}
                elif column in self.score_options:
                    params = {'y_true' : self.recipe.ingredients.y_test,
                              'y_score' : self.predicted_probs[:, 1]}
                else:
                    params = {'y_true' : self.recipe.ingredients.y_test,
                              'y_pred' : self.predictions}
                if column in self.special_metrics:
                    params.update({column : self.special_metrics[column]})
                result = value(**params)
                if column in self.negative_metrics:
                    result = -1 * result
                self.result[column] = result
        self.report.loc[len(self.report)] = self.result
        return self




