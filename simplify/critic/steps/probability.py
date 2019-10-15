"""
.. module:: probability
:synopsis: creates predicted probabilities from machine learning models
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

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
    'gini': self._get_gini_probabilities,
    'log': self._get_log_probabilities,
    'shap': self._get_shap_probabilities}


@dataclass
class Probability(CriticTechnique):
    """Creates predictions from fitted models for out-of-sample data.

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
    name: str = 'probabilities'
    auto_publish: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
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
        return self

