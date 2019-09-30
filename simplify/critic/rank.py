
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.base import SimplePlan, SimpleStep


@dataclass
class Rank(SimplePlan):
    """Creates feature importances for models using a variety of methods.

    Args:

    """
    steps: object = None
    name: str = 'explainer'
    auto_finalize: bool = True

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

    def produce(self):


        return self


@dataclass
class GiniImportances(SimpleStep):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self

    def produce(self, recipe):
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


    def produce(self, recipe):
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

    def _produce_cover(self, recipe):

        return importances

    def _produce_weight(self, recipe):
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