
"""
.. module:: rankers
:synopsis: techniques for calculating feature importances
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
            importances = pd.Series(
                    data = recipe.model.algorithm.feature_importances_,
                    index = features)
            importances.sort_values(ascending = False, inplace = True)
        else:
            importances = None
        return importances


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
        importances = show_weights(
                importance_instance,
                feature_names = recipe.ingredients.columns.keys())
        return importances

@dataclass
class ShapImportances(SimpleTechnique):

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
class BuiltinImportances(SimpleTechnique):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def implement(self, recipe = None, explainer = None):
        return importances