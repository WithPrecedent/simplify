"""
.. module:: rank
:synopsis: calculates feature importances
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from simplify.core.technique import CriticTechnique
from simplify.critic.review import CriticTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'gini': ['self', '_get_sklearn_importances'],
    'permutation': ['eli5.permutation_importance',
                    'get_score_importances'],
    'shap': ['self', '_get_sklearn_importances']}


@dataclass
class Rank(CriticTechnique):
    """Determines feature importances through a variety of techniques.

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
    name: str = 'importances'
    auto_publish: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methos """

    def _get_permutation_importances(self, recipe):
        scorer = self.listify(self.metrics_techniques)[0]
        base_score, score_decreases = self.options[self.technique](
                score_func = scorer,
                x = getattr(recipe.ingredients, 'x_' + self.data_to_review),
                y = getattr(recipe.ingredients, 'y_' + self.data_to_review))
        return np.mean(score_decreases, axis = 'columns')

    def _get_sklearn_importances(self, recipe):
        features = list(getattr(
                recipe.ingredients, 'x_' + self.data_to_review).columns)
        if hasattr(recipe.model.algorithm, 'feature_importances_'):
            importances = pd.Series(
                data = recipe.model.algorithm.feature_importances_,
                index = features)
            return importances.sort_values(ascending = False, inplace = True)
        else:
            return None

    def _get_shap_importances(self, recipe):
        if hasattr(recipe, 'shap_explain.values'):
            return np.abs(recipe.shap_explain.values).mean(0)
        else:
            return None

    """ Core siMpLify Public Methods """

    def draft(self):
        super().draft()
        self.sequence_setting = 'ranking_techniques'
        return self


@dataclass
class RankSelect(CriticTechnique):
    """Uses feature importances for feature reduction in Chef package.

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
    name: str = 'rank'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self