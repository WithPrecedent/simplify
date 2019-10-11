"""
.. module:: explain
:synopsis: explains machine learning results
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import numpy as np

from simplify.core.critic.review import CriticTechnique


@dataclass
class Explain(CriticTechnique):
    """Explains model results.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """

    technique: object = None
    parameters: object = None
    name: str = 'explanations'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
            'eli5': Eli5Explain,
            'shap': ShapExplain,
            'skater': SkaterExplain}
        return self


@dataclass
class Eli5Explain(CriticTechnique):
    """Explains fit model with eli5 package.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """

    technique: object = None
    parameters: object = None
    name: str = 'eli5'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
            'feature': ['eli5', 'explain_prediction_df'],
            }
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
        base_score, score_decreases = get_score_importances(score_func, X, y)
        feature_importances = np.mean(score_decreases, axis=0)
        from eli5 import show_weights
        self.permutation_weights = show_weights(
                self.permutation_importances,
                feature_names = recipe.ingredients.columns.keys())
        return self


@dataclass
class ShapExplain(CriticTechnique):
    """Explains fit model with shap package.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """

    technique: object = None
    parameters: object = None
    name: str = 'shap_explanation'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _set_method(self, recipe):
        if self.technique in self.models:
            self.method = self.options[self.models[self.technique]]
        else:
            self.method = self.options['kernel']
        self.evaluator = self.method(
            model = recipe.model.algorithm,
            data = getattr(recipe.ingredients, 'x_' + self.data_to_review))
        return self

    """ Core siMpLify Methods """

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
        if self.technique != 'none':
            self._set_method()
            setattr(recipe, self.name + '_values', self.evaluator.shap_values(
                    getattr(recipe.ingredients, 'x_' + self.data_to_review)))
            if not hasattr(recipe, self.name):
                setattr(recipe, self.name, [])
            getattr(recipe, self.name).append(self.name)
            if self.method == 'tree':
                setattr(recipe, self.name + '_interactions',
                        self.evaluator.shap_interaction_values(
                                getattr(recipe.ingredients,
                                        'x_' + self.data_to_review)))
            getattr(recipe, self.name).append(
                    getattr(self, self.name + '_interactions'))
        return getattr(recipe, self.name)


@dataclass
class SkaterExplain(CriticTechnique):
    """Explains fit model with skater package.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """

    technique: object = None
    parameters: object = None
    name: str = 'skater'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        return self
