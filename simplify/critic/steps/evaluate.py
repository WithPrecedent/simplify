
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.base import SimpleStep
    

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
        self.permutation_importances = self.options['permutation'](
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
        """Sets up the core attributes of a SklearnEvaluator instance."""
        super().__post_init__()
        return self
    
    def produce(self, recipe):
        self.features = list(self.recipe.ingredients.x_test.columns)
        if hasattr(self.recipe.model.algorithm, 'feature_importances_'):
            self.feature_importances = pd.Series(
                    data = self.recipe.model.algorithm.feature_importances_,
                    index = self.features)
            self.feature_importances.sort_values(ascending = False,
                                                 inplace = True)
        else:
            self.feature_importances = None
        return self
            




@dataclass
class SkaterEvaluator(SimpleStep):

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        return self
