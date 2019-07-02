
from dataclasses import dataclass

import pandas as pd
import sklearn.metrics as met
from eli5 import explain_prediction_df, explain_weights_df, show_prediction
from eli5 import show_weights
from shap import DeepExplainer, KernelExplainer, LinearExplainer, TreeExplainer

from ..cookbook.countertop import Countertop


@dataclass
class Review(Countertop):
    """Computes and stores machine learning experiment results.

    Review creates and stores a results report and other general
    scorers/metrics for machine learning based upon the type of model used in
    the siMpLify package. Users can manually add metrics not already included
    in the metrics dictionary by passing them to Results.add_metric.

    Attributes:
        recipe: an instance of Recipe which contains fit models and transformed
            data. A recipe should only be passed when the class is instanced if
            evaluating an unpickled Cookbook or Recipe. Ordinarily, a recipe
            is passed to Results.evaluate. recipe is overwritten each time a
            new recipe is passed to evaluate_recipe.
    """
    recipe : object = None

    def __post_init__(self):
        self._set_defaults()
        self._set_explainers()
        return self

    def _add_result(self, recipe):
        """Prepares the results of a single recipe application to be added to
        the .report dataframe.
        """
        self.result = pd.Series(index = self.columns_list)
        for column, value in self.columns.items():
            if hasattr(getattr(recipe, value), 'technique'):
                self.result[column] = self._format_step(recipe, value)
            else:
                self.result[column] = getattr(recipe, value)
        for column, value in self.techniques.items():
            if column in self.metrics:
                if column in self.prob_techniques:
                    params = {'y_true' : recipe.ingredients.y_test,
                              'y_prob' : self.predicted_probs[:, 1]}
                elif column in self.score_techniques:
                    params = {'y_true' : recipe.ingredients.y_test,
                              'y_score' : self.predicted_probs[:, 1]}
                else:
                    params = {'y_true' : recipe.ingredients.y_test,
                              'y_pred' : self.predictions}
                if column in self.special_metrics:
                    params.update({column : self.special_metrics[column]})
                result = value(**params)
                if column in self.negative_metrics:
                    result = -1 * result
                self.result[column] = result
        self.report.loc[len(self.report)] = self.result
        return self

    def _check_algorithm(self, step):
        """Returns appropriate algorithm to Review.report."""
        if step.name in ['none', 'all']:
            return 'none'
        else:
            return step.algorithm

    def _class_report(self, recipe):
        self.class_report = met.classification_report(
                recipe.ingredients.y_test,
                self.predictions)
        self.class_report_dict = met.classification_report(
                recipe.ingredients.y_test,
                self.predictions,
                output_dict = True)
        self.class_report_df = pd.DataFrame(self.class_report_dict).transpose()
        return self

    def _confusion(self, recipe):
        self.confusion = met.confusion_matrix(recipe.ingredients.y_test,
                                              self.predictions)
        return self

    def _default_classifier(self):
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
        self.prob_techniques = {'brier_score_loss' : met.brier_score_loss}
        self.score_techniques = {'roc_auc' :  met.roc_auc_score}
        return self

    def _default_clusterer(self):
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
        self.prob_techniques = {}
        self.score_techniques = {}
        return self

    def _default_regressor(self):
        self.techniques = {
                'explained_variance' : met.explained_variance_score,
                'max_error' : met.max_error,
                'absolute_error' : met.absolute_error,
                'mse' : met.mean_squared_error,
                'msle' : met.mean_squared_log_error,
                'mae' : met.median_absolute_error,
                'r2' : met.r2_score}
        self.prob_techniques = {}
        self.score_techniques = {}
        return self

    def _explain(self, recipe):
        for explainer in self._listify(self.explainers):
            explain_package = self.explainer_techniques[explainer]
            explain_package(recipe)
        return self

    def _eli5_explainer(self, recipe):
        return self

    def _feature_summaries(self, recipe):
        self.feature_list = list(recipe.ingredients.x_test.columns)
        if ('svm_' in recipe.model.technique
                or 'baseline' in recipe.model.technique
                or 'logit' in recipe.model.technique):
            self.feature_import = None
        else:
            self.feature_import = pd.Series(
                    data = recipe.model.algorithm.feature_importances_,
                    index = self.feature_list)
            self.feature_import.sort_values(ascending = False,
                                            inplace = True)
        return self

    def _format_step(self, recipe, attribute):
        if getattr(recipe, attribute).technique in ['none', 'all']:
            step_column = getattr(recipe, attribute).technique
        else:
            technique = getattr(recipe, attribute).technique
            parameters = getattr(recipe, attribute).parameters
            step_column = f'{technique}, parameters = {parameters}'
        return step_column

    def _make_predictions(self, recipe):
        """Makes predictions and determines predicted probabilities using
        the model in the recipe passed."""
        self.predictions = recipe.model.algorithm.predict(
                recipe.ingredients.x_test)
        self.predicted_probs = recipe.model.algorithm.predict_proba(
                recipe.ingredients.x_test)
        return self

    def _print_results(self, recipe):
        """Prints to console basic results separate from report."""
        print('These are the results using the', recipe.model.technique,
              'model')
        if recipe.splicer.technique != 'none':
            print('Testing', recipe.splicer.technique, 'predictors')
        print('Confusion Matrix:')
        print(self.confusion)
        print('Classification Report:')
        print(self.class_report)
        return self

    def _set_columns(self, recipe):
        """Sets columns and options for report."""
        self.columns = {'recipe_number' : 'number',
                        'step_order' : 'order',
                        'seed' : 'seed',
                        'validation_set' : 'use_val_set'}
        for step in recipe.order:
            self.columns.update({step : step})
        self.columns_list = list(self.columns.keys())
        self.columns_list.extend(self._listify(self.metrics))
        self.report = pd.DataFrame(columns = self.columns_list)
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
        self.techniques_dict = {'classifier' : self._default_classifier,
                                'regressor' : self._default_regressor,
                                'clusterer' : self._default_clusterer}
        self.techniques_dict[self.model_type]()
        self.techniques.update(self.prob_techniques)
        self.techniques.update(self.score_techniques)
        return self

    def _set_explainers(self):
        """Sets options for explainer(s) chosen by user."""
        self.explainer_techniques = {'shap' : self._shap_explainer,
                                     'eli5' : self._eli5_explainer}
        self.shap_models = {'catboost' : 'tree',
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
                            'xgb' : 'tree'}
        self.shap_techniques = {'deep' : DeepExplainer,
                                'kernel' : KernelExplainer,
                                'linear' : LinearExplainer,
                                'tree' : TreeExplainer}
        return self

    def _shap_explainer(self, recipe):
        """Applies shap explainer to data based upon type of model used."""
        if recipe.model.technique in self.shap_models:
            self.shap_method_type = self.shap_models[recipe.model.technique]
            self.shap_method = self.shap_techniques[self.shap_method_type]
        elif 'baseline_' in recipe.model.technique:
            self.shap_method_type = 'none'
        else:
            self.shap_method_type = 'kernel'
            self.shap_method = KernelExplainer
        data_to_explain = {'train' : recipe.ingredients.x_train,
                           'test' : recipe.ingredients.x_test,
                           'full' : recipe.ingredients.x}
        df = data_to_explain[self.data_to_explain]
        if self.shap_method_type != 'none':
            self.shap_explainer = self.shap_method(
                    model = recipe.model.algorithm,
                    data = recipe.ingredients.x_train)
            self.shap_values = self.shap_explainer.shap_values(df)
            if self.shap_method_type == 'tree':
                self.shap_interactions = (
                        self.shap_explainer.shap_interaction_values(
                                pd.DataFrame(df, columns = df.columns)))
            else:
                self.shap_interactions = None
        return self

    def add_metric(self, name, metric, special_type = None,
                   special_parameters = None, negative_metric = False):
        """Allows user to manually add a metric to report."""
        self.techniques.update({name : metric})
        if special_type in ['probability']:
            self.prob_techniques.update({name : metric})
        elif special_type in ['scorer']:
            self.score_techniques.update({name : metric})
        if special_parameters:
           self.special_metrics.update({name : special_parameters})
        if negative_metric:
           self.special_metrics.append[name]
        return self

    def implement(self, recipe):
        """Evaluates recipe with various tools and prepares report."""
        if self.verbose:
            print('Evaluating recipe')
        if not hasattr(self, 'columns'):
            self._set_columns(recipe)
        self._make_predictions(recipe)
        self._add_result(recipe)
        self._confusion(recipe)
        self._class_report(recipe)
        self._feature_summaries(recipe)
        self._explain(recipe)
        return self