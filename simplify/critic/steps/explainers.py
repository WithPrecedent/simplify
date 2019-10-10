"""
.. module:: explainers
:synopsis: explainers for machine learning results
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.technique import SimpleTechnique


@dataclass
class Eli5Explain(SimpleTechnique):
    """Explains fit model with eli5 package.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'eli5'
    auto_publish: bool = True

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
            'feature': ['eli5', 'explain_prediction_df'],
            'permutation': ['eli5.sklearn', 'PermutationImportance']}
        self.models = {
            'baseline': 'none',
            'catboost': 'specific',
            'decision_tree': 'specific',
            'lasso': 'specific',
            'lasso_lars': 'specific',
            'light_gbm': 'specific',
            'logit': 'specific',
            'ols': 'specific',
            'random_forest': 'specific',
            'ridge': 'specific',
            'svm_linear': 'specific',
            'tensor_flow': 'permutation',
            'torch': 'permutation',
            'xgboost': 'specific'}
        return self

    def implement(self, recipe):
        from eli5 import show_weights
        self.permutation_weights = show_weights(
                self.permutation_importances,
                feature_names = recipe.ingredients.columns.keys())
        return self


@dataclass
class ShapExplain(SimpleTechnique):
    """Explains fit model with shap package.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'shap_explain'
    auto_publish: bool = True

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        self.options = {
            'deep': ['shap', 'DeepExplainer'],
            'kernel': ['shap', 'KernelExplainer'],
            'linear': ['shap', 'LinearExplainer'],
            'tree': ['shap', 'TreeExplainer']}
        self.models = {
            'baseline': 'none',
            'catboost': 'tree',
            'decision_tree': 'tree',
            'lasso': 'linear',
            'lasso_lars': 'linear',
            'light_gbm': 'tree',
            'logit': 'linear',
            'ols': 'linear',
            'random_forest': 'tree',
            'ridge': 'linear',
            'svm_linear': 'linear',
            'tensor_flow': 'deep',
            'torch': 'deep',
            'xgboost': 'tree'}
        return self

    def implement(self, recipe):
        """Applies shap evaluator to data based upon type of model used."""
        if recipe.model.technique in self.shap_models:
            self.shap_method_type = self.shap_models[
                    recipe.model.technique]
            self.shap_method = self.shap_options[self.shap_method_type]
        else:
            self.shap_method_type = 'kernel'
        df = self.options[self.data_to_evaluate]
        if self.shap_method_type != 'none':
            self.shap_evaluator = self.shap_method(
                    model = recipe.model.algorithm,
                    data = recipe.ingredients.x_train)
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
class SkaterExplain(SimpleTechnique):
    """Explains fit model with skater package.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'skater'
    auto_publish: bool = True

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        return self
