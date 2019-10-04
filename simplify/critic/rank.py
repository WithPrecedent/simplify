"""
.. module:: rank
:synopsis: calculates feature importances
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.plan import SimplePlan
from simplify.core.step import SimpleStep


@dataclass
class Rank(SimplePlan):
    """Determines feature importances through a variety of techniques.

    Args:
        steps(dict(str: SimpleStep)): names and related SimpleStep classes for
            explaining data analysis models.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
    """

    steps: object = None
    name: str = 'ranker'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        self.options = {
                'gini': GiniImportances,
                'permutation': PermutationImportances,
                'shap': ShapImportances,
                'builtin': BuiltinImportances}
        self.step_iterable = 'feature_importances'
        self.idea_setting = 'feature_importance_technique'
        return self

    def implement(self, recipe = None, explainer = None):
        for name in self.options.keys():
            if name in getattr(self, self.idea_setting):
                importances = getattr(self, name).implement(
                        recipe = recipe,
                        explainer = explainer)
                getattr(self, self.step_iterable).update({name: importances})
        return self


@dataclass
class GiniImportances(SimpleStep):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def implement(self, recipe = None, explainer = None):
        features = list(recipe.ingredients.x_test.columns)
        if hasattr(recipe.model.algorithm, 'feature_importances_'):
            importances = pd.Series(
                    data = recipe.model.algorithm.feature_importances_,
                    index = features)
            importances.sort_values(ascending = False, inplace = True)
        else:
            importances = None
        return importances


@dataclass
class PermutationImportances(SimpleStep):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {'eli5': ['eli5.sklearn', 'PermutationImportance']}
        return self

    def implement(self, recipe = None, explainer = None):

        from eli5 import show_weights

        importance_instance = self.options['eli5'](
                estimator = recipe.model.algorithm,
                random_state = self.seed)
        importance_instance.fit(
                recipe.ingredients.x_test,
                recipe.ingredients.y_test)
        importances = show_weights(
                importance_instance,
                feature_names = recipe.ingredients.columns.keys())
        return importances

@dataclass
class ShapImportances(SimpleStep):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def implement(self, recipe = None, explainer = None):
        importances = np.abs(explainer.shap_values).mean(0)
        return importances


@dataclass
class BuiltinImportances(SimpleStep):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def implement(self, recipe = None, explainer = None):
        return importances

@dataclass
class RankSelect(SimpleStep):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self