
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.base import SimpleStep


@dataclass
class Evaluate(SimpleStep):
    """Core class for evaluating the results of data analysis produceed by
    the siMpLify Cookbook.

    """
    techniques : object = None
    name : str = 'evaluator'
    auto_finalize : bool = True
    
    def __post_init__(self):
        """Sets up the core attributes of an Evaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        self.options = {'eli5' : Eli5Evaluator,
                        'shap' : ShapEvaluator,
                        'skater' : SkaterEvaluator,
                        'sklearn' : SklearnEvaluator}
        self.checks = ['idea']
        return self         

@dataclass
class Eli5Evaluator(SimpleStep):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def draft(self):

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

    def produce(self):
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
class ShapEvaluator(SimpleStep):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
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

    def produce(self):
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
        self.feature_importances = np.abs(self.shap_values).mean(0)
        return self

@dataclass
class SklearnEvaluator(SimpleStep):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def _confusion_matrix(self):
        self.confusion = metrics.confusion_matrix(
                self.recipe.ingredients.y_test, self.predictions)
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

    def draft(self):
        getattr(self, '_default_' + self.model_type)()
        self.special_metrics = {
                'fbeta' : {'beta' : 1},
                'f1_weighted' : {'average' : 'weighted'},
                'precision_weighted' : {'average' : 'weighted'},
                'recall_weighted' : {'average' : 'weighted'}}
        self.negative_metrics = ['brier_loss_score', 'neg_log_loss',
                                 'zero_one']
        return self

    def finalize(self):
        self.options.update(self.prob_options)
        self.options.update(self.score_options)
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

@dataclass
class SkaterEvaluator(SimpleStep):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        return self
