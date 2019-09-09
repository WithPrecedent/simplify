
from dataclasses import dataclass

import pandas as pd
from sklearn import metrics

from simplify.cookbook.cookbook_step import CookbookStep
from simplify.core.base import SimpleClass


@dataclass
class Review(SimpleClass):
    """Computes and stores machine learning experiment results.

    Review creates and stores a results report and other general
    scorers/metrics for machine learning based upon the type of model used in
    the siMpLify package. Users can manually add metrics not already included
    in the metrics dictionary by passing them to Results.add_metric.

    Attributes:
        name: a string designating the name of the class which should be
            identical to the section of the menu with relevant settings.
        auto_prepare: sets whether to automatically call the prepare method
            when the class is instanced. If you do not plan to make any
            adjustments to the options or metrics beyond the menu, this option
            should be set to True. If you plan to make such changes, prepare
            should be called when those changes are complete.
    """
    name : str = 'review'
    auto_prepare : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def _check_algorithm(self, step):
        """Returns appropriate algorithm to the report attribute."""
        if step.technique in ['none', 'all']:
            return step.technique
        else:
            return step.algorithm

    def _cluster_report(self):
        return self

    def _format_step(self, attribute):
        if getattr(self.recipe, attribute).technique in ['none', 'all']:
            step_column = getattr(self.recipe, attribute).technique
        else:
            technique = getattr(self.recipe, attribute).technique
            parameters = getattr(self.recipe, attribute).parameters
            step_column = f'{technique}, parameters = {parameters}'
        return step_column

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

    def _regressor_report(self):
        return self

    def _set_columns(self):
        """Sets columns and options for report."""
        self.columns = {'recipe_number' : 'number',
                        'options' : 'techniques',
                        'seed' : 'seed',
                        'validation_set' : 'val_set'}
        for step in self.recipe.techniques:
            self.columns.update({step : step})
        self.columns_list = list(self.columns.keys())
        self.columns_list.extend(self.listify(self.metrics))
        self.report = pd.DataFrame(columns = self.columns_list)
        return self

    def _set_defaults(self):
        """Sets default metrics for scores dataframe based upon the type of
        model used.
        """
        self.options = {'eli5' : Eli5Evaluator,
                        'lime' : LimeEvaluator,
                        'shap' : ShapEvaluator,
                        'sklearn' : SklearnEvaluator,
                        'tree' : TreeEvaluator}
        return self

    def _set_techniques(self):
        if self.evaluators == 'all':
            self.techniques = list(self.options.keys())
        else:
            self.techniques = self.listify(self.evalutors)
        return self


    def prepare(self):
        return self

    def start(self, recipe):
        """Evaluates recipe with various tools and prepares report."""
        if self.verbose:
            print('Evaluating recipe')
        self.recipe = recipe
        if not hasattr(self, 'columns'):
            self._set_columns()
        self._create_predictions()
        self._add_result()
        self._confusion_matrix()
        getattr(self, '_' + self.model_type + '_report')()
        self._feature_summaries()
        for evaluator in self.listify(self.evaluators):
            evaluate_package = self.evaluator_options[evaluator]
            evaluate_package()
        return self


@dataclass
class Evaluator(SimpleClass):

    def __post_init__(self):
        """Sets up the core attributes of an Evaluator instance."""
        super().__post_init__()
        return self

    def _set_defaults(self):

        self.options = {}
        return self

@dataclass
class Eli5Evaluator(Evaluator):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def _set_defaults(self):

        from eli5 import explain_prediction_df, explain_weights_df
        from eli5.sklearn import PermutationImportance

        self.options = {'specific' : explain_prediction_df,
                        'permutation' : PermutationImportance}
        self.models = {'baseline' : 'none',
                       'catboost' : 'specific',
                       'decision_tree' : 'specific',
                       'lasso' : 'specific',
                       'lasso_lars' : 'specific',
                       'light_gbm' : 'specific',
                       'logit' : 'specific',
                       'ols' : 'specific',
                       'random_forest' : 'specific',
                       'ridge' : 'specific',
                       'svm_linear' : 'specific',
                       'tensor_flow' : 'permutation',
                       'torch' : 'permutation',
                       'xgboost' : 'specific'}
        return self

    def start(self):
        self.permutation_importances = PermutationImportance(
                self.recipe.model.algorithm,
                random_state = self.seed).fit(
                        self.recipe.ingredients.x_test,
                        self.recipe.ingredients.y_test)
        self.permutation_weights = show_weights(
                self.permutation_importances,
                feature_names = self.recipe.ingredients.columns.keys())
        return self


@dataclass
class LimeEvaluator(Evaluator):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def _set_defaults(self):
        return self


@dataclass
class ShapEvaluator(Evaluator):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def _set_defaults(self):
        from shap import (DeepExplainer, KernelExplainer, LinearExplainer,
                          TreeExplainer)

        self.options = {'deep' : DeepExplainer,
                        'kernel' : KernelExplainer,
                        'linear' : LinearExplainer,
                        'tree' : TreeExplainer}
        self.models = {'baseline' : 'none',
                       'catboost' : 'tree',
                       'decision_tree' : 'tree',
                       'lasso' : 'linear',
                       'lasso_lars' : 'linear',
                       'light_gbm' : 'tree',
                       'logit' : 'linear',
                       'ols' : 'linear',
                       'random_forest' : 'tree',
                       'ridge' : 'linear',
                       'svm_linear' : 'linear',
                       'tensor_flow' : 'deep',
                       'torch' : 'deep',
                       'xgboost' : 'tree'}
        return self

    def start(self):
        """Applies shap evaluator to data based upon type of model used."""
        if self.recipe.model.technique in self.shap_models:
            self.shap_method_type = self.shap_models[
                    self.recipe.model.technique]
            self.shap_method = self.shap_options[self.shap_method_type]
        else:
            self.shap_method_type = 'kernel'

        df = self.options[self.data_to_evaluate]
        if self.shap_method_type != 'none':
            self.shap_evaluator = self.shap_method(
                    model = self.recipe.model.algorithm,
                    data = self.recipe.ingredients.x_train)
            self.shap_values = self.shap_evaluator.shap_values(df)
            if self.shap_method_type == 'tree':
                self.shap_interactions = (
                        self.shap_evaluator.shap_interaction_values(
                                pd.DataFrame(df, columns = df.columns)))
            else:
                self.shap_interactions = None
        return self

@dataclass
class SklearnEvaluator(Evaluator):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
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
        self.confusion = metrics.confusion_matrix(self.recipe.ingredients.y_test,
                                                  self.predictions)
        return self

    def _create_predictions(self):
        """Makes predictions and determines predicted probabilities using
        the model in the recipe passed."""
        if hasattr(self.recipe.model.algorithm, 'predict'):
            self.predictions = self.recipe.model.algorithm.predict(
                    self.recipe.ingredients.x_test)
        if hasattr(self.recipe.model.algorithm, 'predict_proba'):
            self.predicted_probs = self.recipe.model.algorithm.predict_proba(
                    self.recipe.ingredients.x_test)
        return self

    def _default_classifier(self):
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

    def _feature_summaries(self):
        self.feature_list = list(self.recipe.ingredients.x_test.columns)
        if ('svm_' in self.recipe.model.technique
                or 'baseline' in self.recipe.model.technique
                or 'logit' in self.recipe.model.technique
                or 'tensor_flow' in self.recipe.model.technique):
            self.feature_import = None
        else:
            self.feature_import = pd.Series(
                    data = self.recipe.model.algorithm.feature_importances_,
                    index = self.feature_list)
            self.feature_import.sort_values(ascending = False,
                                            inplace = True)
        return self

    def _set_defaults(self):
        getattr(self, '_default_' + self.model_type)()
        self.special_metrics = {
                'fbeta' : {'beta' : 1},
                'f1_weighted' : {'average' : 'weighted'},
                'precision_weighted' : {'average' : 'weighted'},
                'recall_weighted' : {'average' : 'weighted'}}
        self.negative_metrics = ['brier_loss_score', 'neg_log_loss',
                                 'zero_one']
        return self

    def add_metric(self, name, metric, special_type = None,
                   special_parameters = None, negative_metric = False):
        """Allows user to manually add a metric to report."""
        self.options.update({name : metric})
        if special_type in ['probability']:
            self.prob_options.update({name : metric})
        elif special_type in ['scorer']:
            self.score_options.update({name : metric})
        if special_parameters:
           self.special_metrics.update({name : special_parameters})
        if negative_metric:
           self.special_metrics.append[name]
        return self

    def prepare(self):
        self.options.update(self.prob_options)
        self.options.update(self.score_options)
        return self

    def start(self):
        """Prepares the results of a single recipe application to be added to
        the .report dataframe.
        """
        self.result = pd.Series(index = self.columns_list)
        for column, value in self.columns.items():
            if isinstance(getattr(self.recipe, value), CookbookStep):
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

@dataclass
class TreeEvaluator(Evaluator):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def _set_defaults(self):
        return self


