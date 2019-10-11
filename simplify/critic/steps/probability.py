"""
.. module:: probability
:synopsis: creates predicted probabilities from machine learning models
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.iterable import SimpleIterable
from simplify.core.technique import SimpleTechnique


@dataclass
class Probability(SimpleIterable):
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

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
            'gini': GiniProbabilities,
            'log': LogProbabilities,
            'shap': ShapProbabilities}
        return self


@dataclass
class GiniProbabilities(SimpleTechnique):
    """Estimates probabilities of outcomes based upon fit model.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'gini_probabilities'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def implement(self, recipe):
        """Makes predictions from fitted model.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predicted probabilities from fitted model on test data.
        """
        if hasattr(self.recipe.model.algorithm, 'predict_proba'):
            return self.recipe.model.algorithm.predict_proba(
                    self.recipe.ingredients.x_test)
        else:
            if self.verbose:
                print('predict_proba method does not exist for',
                    self.recipe.model.technique.name)
            return None


@dataclass
class LogProbabilities(SimpleTechnique):
    """Estimates log probabilities of outcomes based upon fit model.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'log_probabilities'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def implement(self, recipe):
        """Makes predictions from fitted model.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predicted probabilities from fitted model on test data.
        """
        if hasattr(self.recipe.model.algorithm, 'predict_log_proba'):
            return self.recipe.model.algorithm.predict_log_proba(
                    self.recipe.ingredients.x_test)
        else:
            if self.verbose:
                print('predict_log_proba method does not exist for',
                    self.recipe.model.technique.name)
            return None


@dataclass
class ShapProbabilities(SimpleTechnique):
    """Estimates probabilities of outcomes based upon fit model using SHAP
    values.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'shap_probabilities'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def implement(self, recipe):
        """Makes predictions from fitted model.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predicted probabilities from fitted model on test data.
        """
        return None