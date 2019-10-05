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

from simplify.core.iterables import SimplePlan
from simplify.core.technique import SimpleTechnique


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
                'gini': ['simplify.critic.steps.rankers', 'GiniImportances'],
                'permutation': ['simplify.critic.steps.rankers', 
                                'PermutationImportances'],
                'shap': ['simplify.critic.steps.rankers', 'ShapImportances'],
                'builtin': ['simplify.critic.steps.rankers', 
                            'BuiltinImportances']}
        self.iterable = 'feature_importances'
        self.idea_setting = 'feature_importance_technique'
        return self

    def implement(self, ingredients = None, recipes = None, explainers = None):
        for name in self.options.keys():
            if name in getattr(self, self.idea_setting):
                importances = getattr(self, name).implement(
                        recipe = recipes,
                        explainer = explainers)
                getattr(self, self.iterable).update({name: importances})
        return self


@dataclass
class RankSelect(SimpleTechnique):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self