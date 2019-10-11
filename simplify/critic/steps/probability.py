"""
.. module:: probability
:synopsis: creates predicted probabilities from machine learning models
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.critic.review import CriticTechnique


@dataclass
class Probability(CriticTechnique):
    """Creates predictions from fitted models for out-of-sample data.

    Args:
        steps(dict(str: SimpleTechnique)): names and related SimpleTechnique
            classes for creating predictions data analysis models.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_implement (bool): whether to call the 'implement' method when the
            class is instanced.
    """
    steps: object = None
    name: str = 'probabilities'
    auto_publish: bool = True
    auto_implement: bool = False

    def __post_init__(self):
        self.idea_sections = ['critic']
        super().__post_init__()
        return self

    """ Private Methos """

    def _get_gini_probabilities(self, recipe):
        """Estimates probabilities of outcomes from fitted model with gini
        method.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predictions from fitted model on test data.

        """
        if hasattr(recipe.model.algorithm, 'predict_proba'):
            return recipe.model.algorithm.predict_proba(
                getattr(recipe.ingredients, 'x_' + self.data_to_review))[1]
        else:
            if self.verbose:
                print('predict_proba method does not exist for',
                      recipe.model.technique.name)
            return None

    def _get_log_probabilities(self, recipe):
        """Estimates log probabilities of outcomes from fitted model.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predictions from fitted model on test data.

        """
        if hasattr(recipe.model.algorithm, 'predict_log_proba'):
            return recipe.model.algorithm.predict_log_proba(
                getattr(recipe.ingredients, 'x_' + self.data_to_review))[1]
        else:
            if self.verbose:
                print('predict_log_proba method does not exist for',
                      recipe.model.technique.name)
            return None

    def _get_shap_probabilities(self, recipe):
        """Estimates probabilities of outcomes from fitted model with shap
        values.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predictions from fitted model on test data.

        """
        return None

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
            'gini': self._get_gini_probabilities,
            'log': self._get_log_probabilities,
            'shap': self._get_shap_probabilities}
        return self

