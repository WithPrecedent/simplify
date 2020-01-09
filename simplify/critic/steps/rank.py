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

from simplify.core.typesetter import CriticTechnique
from simplify.critic.collection import CriticTechnique


@dataclass
class Rank(CriticTechnique):
    """[summary]

    Args:
        CriticTechnique ([type]): [description]

    Returns:
        [type]: [description]
    """
    step: object = None
    parameters: object = None
    name: str = 'importances'
    auto_draft: bool = True
    auto_publish: bool = False

    def __post_init__(self) -> None:
        self.idea_sections = ['critic']
        super().__post_init__()
        return self

    """ Private Methods """

    def _get_permutation_importances(self, recipe):
        scorer = listify(self.metrics_steps)[0]
        base_score, score_decreases = self.library[self.step](
                score_func = scorer,
                x = getattr(recipe.ingredients, 'x_' + self.data_to_review),
                y = getattr(recipe.ingredients, 'y_' + self.data_to_review))
        return np.mean(score_decreases, axis = 'columns')

    def _get_gini_importances(self, recipe):
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

    def _get_eli5_importances(self, recipe):
        base_score, score_decreases = get_score_importances(score_func, X, y)
        feature_importances = np.mean(score_decreases, axis=0)
        from eli5 import show_weights
        self.permutation_weights = show_weights(
                self.permutation_importances,
                feature_names = recipe.ingredients.columns.keys())
        return self

    """ Core siMpLify Public Methods """

    def draft(self) -> None:
        super().draft()
        self.steps_setting = 'ranking_steps'
        return self


@dataclass
class RankSelect(CriticTechnique):
    """Uses feature importances for feature reduction in Chef package.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """

    step: object = None
    parameters: object = None
    name: str = 'rank'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        self._options = Contents(options = {}
        return self