"""
.. module:: rank
:synopsis: calculates feature importances
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

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
        auto_read (bool): whether to call the 'read' method when the class
            is instanced.
    """
    steps: object = None
    name: str = 'ranker'
    auto_publish: bool = True
    auto_read: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {
                'gini': GiniImportances,
                'permutation': PermutationImportances,
                'shap': ShapImportances,
                'builtin': BuiltinImportances}
        return self

    def read(self):


        return self


@dataclass
class GiniImportances(SimpleStep):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def read(self, recipe):
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


    def read(self, recipe):
        importance_instance = PermutationImportance(
                estimator = recipe.model.algorithm,
                random_state = self.seed)
        importance_instance.fit(
                recipe.ingredients.x_test,
                recipe.ingredients.y_test)
        self.permutation_weights = show_weights(
                self.permutation_importances,
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


@dataclass
class BuiltinImportances(SimpleStep):

    def __post_init__(self):
        super().__post_init__()
        return self

    def _read_cover(self, recipe):

        return importances

    def _read_weight(self, recipe):
        return importances

    def draft(self):
        self.options = {}
        return self

@dataclass
class RankSelect(SimpleStep):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self