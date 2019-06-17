
from dataclasses import dataclass

import pandas as pd
import sklearn.metrics as met

from ..implements.implement import Implement


@dataclass
class Review(Implement):
    """Computes, stores, and exports machine learning experiment results.

    Review creates and stores a results table and other general
    scorers/metrics for machine learning based upon the type of model used in
    the siMpLify package. Users can manually add metrics not already included
    in the metrics dictionary by passing them to Results.include.

    Attributes:
        recipe: an instance of Recipe which contains fit models and transformed
            data. A recipe need not be passed when class is instanced. It can
            be passed to Results.add directly.
    """
    recipe : object = None

    def __post_init__(self):
        super().__post_init__()
        self.settings.localize(instance = self, sections = ['review_params'])
        self.step_columns = ['recipe_number', 'step_order', 'scale',
                             'split', 'encode', 'mix', 'cleave', 'sample',
                             'reduce', 'model', 'custom', 'seed',
                             'validation_set']
        self.columns = self.step_columns
        self._set_defaults()
        self.columns.extend(self._listify(self.metrics))
        self.table = pd.DataFrame(columns = self.columns)
        return self

    def _set_defaults(self):
        """Sets default metrics for scores dataframe based upon the type of
        model used.
        """
        self.special_metrics = {
                'fbeta' : {'beta' : 1},
                'f1_weighted' : {'average' : 'weighted'},
                'precision_weighted' : {'average' : 'weighted'},
                'recall_weighted' : {'average' : 'weighted'}}
        self.negative_metrics = ['brier_loss_score', 'neg_log_loss',
                                 'zero_one']
        if self.model_type in ['classifier']:
            self.techniques = {
                    'accuracy' : met.accuracy_score,
                    'balanced_accuracy' : met.balanced_accuracy_score,
                    'f1' : met.f1_score,
                    'f1_weighted' : met.f1_score,
                    'fbeta' : met.fbeta_score,
                    'hamming' : met.hamming_loss,
                    'jaccard' : met.jaccard_similarity_score,
                    'matthews_corrcoef' : met.matthews_corrcoef,
                    'neg_log_loss' :  met.log_loss,
                    'precision' :  met.precision_score,
                    'precision_weighted' :  met.precision_score,
                    'recall' :  met.recall_score,
                    'recall_weighted' :  met.recall_score,
                    'zero_one' : met.zero_one_loss}
            self.prob_options = {'brier_score_loss' : met.brier_score_loss}
            self.score_options = {'roc_auc' :  met.roc_auc_score}
        elif self.model_type in ['regressor']:
            self.techniques = {
                    'explained_variance' : met.explained_variance_score,
                    'max_error' : met.max_error,
                    'absolute_error' : met.absolute_error,
                    'mse' : met.mean_squared_error,
                    'msle' : met.mean_squared_log_error,
                    'mae' : met.median_absolute_error,
                    'r2' : met.r2_score}
            self.prob_options = {}
            self.score_options = {}
        elif self.model_type in ['clusterer']:
            self.techniques = {
                    'adjusted_mutual_info' : met.adjusted_mutual_info_score,
                    'adjusted_rand' : met.adjusted_rand_score,
                    'calinski' : met.calinski_harabasz_score,
                    'davies' : met.davies_bouldin_score,
                    'completeness' : met.completeness_score,
                    'contingency_matrix' : met.cluster.contingency_matrix,
                    'fowlkes' : met.fowlkes_mallows_score,
                    'h_completness' : met.homogeneity_completeness_v_measure,
                    'homogeniety' : met.homogeneity_score,
                    'mutual_info' : met.mutual_info_score,
                    'norm_mutual_info' : met.normalized_mutual_info_score,
                    'silhouette' : met.silhouette_score,
                    'v_measure' : met.v_measure_score}
            self.prob_options = {}
            self.score_options = {}
        self.techniques.update(self.prob_options)
        self.techniques.update(self.score_options)
        return self

    def add(self, recipe):
        self.recipe = recipe
        return self

    def add_metric(self, name, metric, special_type = None,
                   special_parameters = None, negative_metric = False):
        """
        Allows user to manually add a metric to the scores dataframe.
        """
        self.techniques.update({name : metric})
        if special_type in ['probability']:
            self.prob_options.update({name : metric})
        elif special_type in ['scorer']:
            self.score_options.update({name : metric})
        if special_parameters:
           self.special_metrics.update({name : special_parameters})
        if negative_metric:
           self.special_metrics.append[name]
        self._set_options()
        self.table = pd.DataFrame(columns = self.columns)
        return self

    def load(self, import_folder = None, file_name = 'results_table',
             file_format = 'csv', import_path = '', encoding = 'windows-1252',
             message = 'Importing results'):
        """Loads results scores file from disc. This method is used if the user
        wants to reconstruct recipes or cookbooks from past experiments.
        """
        if self.verbose:
            print(message)
        if import_path:
            results_path = import_path
        elif self.results_path:
            results_path = self.results_path
        else:
            results_path = self.pantry.make_path(folder = import_folder,
                                                file_name = file_name,
                                                file_type = file_format)
        self.table = self.pantry.load(results_path,
                                     encoding = encoding)
        return self

    def save(self, export_path = None, file_name = 'results_table',
             file_format = 'csv', encoding = 'windows-1252',
             float_format = '%.4f', message = 'Exporting results'):
        """
        Exports results scores to disc.
        """
        if not export_path:
            export_path = self.pantry.make_path(folder = self.pantry.results,
                                               name = file_name,
                                               file_type = file_format)
        self.table.to_csv(export_path,
                          encoding = encoding,
                          float_format = float_format)
#        self.pantry.save(df = self.table,
#                        export_path = export_path,
#                        encoding = encoding,
#                        float_format = float_format,
#                        message = message)
        return self