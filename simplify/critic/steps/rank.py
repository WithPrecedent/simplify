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

from simplify.core.iterable import SimpleIterable
from simplify.core.technique import SimpleTechnique


@dataclass
class Rank(SimpleIterable):
    """Determines feature importances through a variety of techniques.

    Args:
        steps(dict(str: SimpleTechnique)): names and related SimpleTechnique classes for
            explaining data analysis models.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
    """

    steps: object = None
    name: str = 'rank'
    auto_publish: bool = True

    def __post_init__(self):
        self.idea_sections = ['critic']
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        self.options = {
                'gini': ['simplify.critic.steps.rankers', 'GiniImportances'],
                'permutation': ['simplify.critic.steps.rankers',
                                'PermutationImportances'],
                'shap': ['simplify.critic.steps.rankers', 'ShapImportances'],
                'builtin': ['simplify.critic.steps.rankers',
                            'BuiltinImportances']}
        self.sequence_setting = 'importance_techniques'
        self.return_variables = ['importances']
        return self

@dataclass
class RankSelect(SimpleTechnique):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self


from simplify.core.technique import SimpleTechnique


@dataclass
class GiniImportances(SimpleTechnique):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def implement(self, recipe = None, explainer = None):
        features = list(recipe.ingredients.x_test.columns)
        if hasattr(recipe.model.algorithm, 'feature_importances_'):
            self.importances = pd.Series(
                    data = recipe.model.algorithm.feature_importances_,
                    index = features)
            self.importances.sort_values(ascending = False, inplace = True)
        else:
            self.importances = None
        return self


@dataclass
class PermutationImportances(SimpleTechnique):

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
        self.importances = show_weights(
                importance_instance,
                feature_names = recipe.ingredients.columns.keys())
        return self

@dataclass
class ShapImportances(SimpleTechnique):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def implement(self, recipe = None, explainer = None):
        self.importances = np.abs(explainer.shap_values).mean(0)
        return self


@dataclass
class BuiltinImportances(SimpleTechnique):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def implement(self, recipe = None, explainer = None):
        self.importances = None
        return self