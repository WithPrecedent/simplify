"""
.. module:: critic steps
:synopsis: default steps for chef subpackage
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections import namedtuple

fields = [
    'name', 'module', 'algorithm', 'default', 'required',
    'runtime_parameters', 'selected', 'conditional_parameters',
    'data_dependent']
Technique = namedtuple('step', fields, default = (None,) * len(fields))

""" Explanation Techniques """

eli5_explanation = Technique(
    name = 'eli5_explanation',
    module = 'eli5',
    algorithm = 'explain_prediction_df')
shap_deep_explanation = Technique(
    name = 'shap_explanation',
    module = 'shap',
    algorithm = 'DeepExplainer')
shap_kernel_explanation = Technique(
    name = 'shap_explanation',
    module = 'shap',
    algorithm = 'KernelExplainer')
shap_linear_explanation = Technique(
    name = 'shap_explanation',
    module = 'shap',
    algorithm = 'LinearExplainer')
shap_tree_explanation = Technique(
    name = 'shap_explanation',
    module = 'shap',
    algorithm = 'TreeExplainer')

""" Prediction Techniques """

prediction_gini = Technique(
    name = 'gini_predictions',
    module = 'self',
    algorithm = '_get_gini_predictions')
prediction_shap = Technique(
    name = 'shap_predictions',
    module = 'self',
    algorithm = '_get_shap_predictions')

""" Probability Teachniques """

probability_gini = Technique(
    name = 'gini_predicted_probabilities',
    module = 'self',
    algorithm = '_get_gini_probabilities')
probability_log = Technique(
    name = 'log_predicted_probabilities',
    module = 'self',
    algorithm = '_get_log_probabilities')
probability_shap = Technique(
    name = 'shap_predicted_probabilities',
    module = 'self',
    algorithm = '_get_shap_probabilities')

""" Ranking Techniques """

rank_gini = Technique(
    name = 'gini_importances',
    module = 'self',
    algorithm = '_get_gini_importances')
rank_eli5 = Technique(
    name = 'eli5_importances',
    module = 'self',
    algorithm = '_get_permutation_importances')
rank_permutation = Technique(
    name = 'permutation_importances',
    module = 'self',
    algorithm = '_get_eli5_importances')
rank_shap = Technique(
    name = 'shap_importances',
    module = 'self',
    algorithm = '_get_shap_importances')

""" Metrics Techniques """

def _get_brier_score_loss_parameters(self, parameters, recipe = None):
    if self.step in 'brier_score_loss':
        parameters = {
            'y_true': getattr(recipe.ingredients,
                                'y_' + self.data_to_review),
            'y_prob': recipe.probabilities[:, 1)
    elif self.step in ['roc_auc']:
            parameters = {
                'y_true': getattr(recipe.ingredients,
                                'y_' + self.data_to_review),
                'y_score': recipe.probabilities[:, 1)
    return parameters

metrics_accuracy = Technique(
    name = 'accuracy',
    module = 'sklearn.metrics',
    algorithm = 'accuracy_score')
metrics_adjusted_mutual_info = Technique(
    name = 'adjusted_mutual_info_score',
    module = 'sklearn.metrics',
    algorithm = 'adjusted_mutual_info')
metrics_adjusted_rand = Technique(
    name = 'adjusted_rand',
    module = 'sklearn.metrics',
    algorithm = 'adjusted_rand_score')
metrics_balanced_accuracy = Technique(
    name = 'balanced_accuracy',
    module = 'sklearn.metrics',
    algorithm = 'balanced_accuracy_score')
metrics_brier_score_loss = Technique(
    name = 'brier_score_loss',
    module = 'sklearn.metrics',
    algorithm = 'brier_score_loss')
metrics_calinski = Technique(
    name = 'calinski_harabasz',
    module = 'sklearn.metrics',
    algorithm = 'calinski_harabasz_score')
metrics_davies = Technique(
    name = 'davies_bouldin',
    module = 'sklearn.metrics',
    algorithm = 'davies_bouldin_score')
metrics_completeness = Technique(
    name = 'completeness',
    module = 'sklearn.metrics',
    algorithm = 'completeness_score')
metrics_contingency_matrix = Technique(
    name = 'contingency_matrix',
    module = 'sklearn.metrics',
    algorithm = 'cluster.contingency_matrix')
metrics_explained_variance = Technique(
    name = 'explained_variance',
    module = 'sklearn.metrics',
    algorithm = 'explained_variance_score')
metrics_f1 = Technique(
    name = 'f1',
    module = 'sklearn.metrics',
    algorithm = 'f1_score')
metrics_f1_weighted = Technique(
    name = 'f1_weighted',
    module = 'sklearn.metrics',
    algorithm = 'f1_score',
    required = {'average': 'weighted'})
metrics_fbeta = Technique(
    name = 'fbeta',
    module = 'sklearn.metrics',
    algorithm = 'fbeta_score',
    required = {'beta': 1})
metrics_fowlkes = Technique(
    name = 'fowlkes_mallows',
    module = 'sklearn.metrics',
    algorithm = 'fowlkes_mallows_score')
metrics_hamming = Technique(
    name = 'hamming_loss',
    module = 'sklearn.metrics',
    algorithm = 'hamming_loss')
metrics_h_completness = Technique(
    name = 'homogeneity_completeness',
    module = 'sklearn.metrics',
    algorithm = 'homogeneity_completeness_v_measure')
metrics_homogeniety = Technique(
    name = 'homogeneity',
    module = 'sklearn.metrics',
    algorithm = 'homogeneity_score')
metrics_jaccard = Technique(
    name = 'jaccard_similarity',
    module = 'sklearn.metrics',
    algorithm = 'jaccard_similarity_score')
metrics_mae = Technique(
    name = 'median_absolute_error',
    module = 'sklearn.metrics',
    algorithm = 'median_absolute_error')
metrics_matthews_corrcoef = Technique(
    name = 'matthews_correlation_coefficient',
    module = 'sklearn.metrics',
    algorithm = 'matthews_corrcoef')
metrics_max_error = Technique(
    name = 'max_error',
    module = 'sklearn.metrics',
    algorithm = 'max_error')
metrics_mean_absolute_error = Technique(
    name = 'mean_absolute_error',
    module = 'sklearn.metrics',
    algorithm = 'mean_absolute_error')
metrics_mse = Technique(
    name = 'mean_squared_error',
    module = 'sklearn.metrics',
    algorithm = 'mean_squared_error')
metrics_msle = Technique(
    name = 'mean_squared_log_error',
    module = 'sklearn.metrics',
    algorithm = 'mean_squared_log_error')
metrics_mutual_info = Technique(
    name = 'mutual_info_score',
    module = 'sklearn.metrics',
    algorithm = 'mutual_info_score')
metrics_log_loss = Technique(
    name = 'log_loss',
    module = 'sklearn.metrics',
    algorithm = 'log_loss')
metrics_norm_mutual_info = Technique(
    name = 'normalized_mutual_info',
    module = 'sklearn.metrics',
    algorithm = 'normalized_mutual_info_score')
metrics_precision = Technique(
    name = 'precision',
    module = 'sklearn.metrics',
    algorithm = 'precision_score')
metrics_precision_weighted = Technique(
    name = 'precision_weighted',
    module = 'sklearn.metrics',
    algorithm = 'precision_score',
    required = {'average': 'weighted'})
metrics_r2 = Technique(
    name = 'r2',
    module = 'sklearn.metrics',
    algorithm = 'r2_score')
metrics_recall = Technique(
    name = 'recall',
    module = 'sklearn.metrics',
    algorithm = 'recall_score')
metrics_recall_weighted = Technique(
    name = 'recall_weighted',
    module = 'sklearn.metrics',
    algorithm = 'recall_score',
    required = {'average': 'weighted'})
metrics_roc_auc = Technique(
    name = 'roc_auc',
    module = 'sklearn.metrics',
    algorithm = 'roc_auc_score')
metrics_silhouette = Technique(
    name = 'silhouette',
    module = 'sklearn.metrics',
    algorithm = 'silhouette_score')
metrics_v_measure = Technique(
    name = 'v_measure',
    module = 'sklearn.metrics',
    algorithm = 'v_measure_score')
metrics_zero_one = Technique(
    name = 'zero_one',
    module = 'sklearn.metrics',
    algorithm = 'zero_one_loss')

""" Report Techniques """

report_classification = Technique(
    name = 'classification_report',
    module = 'sklearn.metrics',
    algorithm = 'classification_report')
report_confusion = Technique(
    name = 'confusion_matrix',
    module = 'sklearn.metrics',
    algorithm = 'confusion_matrix')