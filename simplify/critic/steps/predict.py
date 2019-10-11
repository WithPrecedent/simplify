"""
.. module:: predict
:synopsis: creates predictions from machine learning models
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.critic.review import CriticTechnique


@dataclass
class Predict(CriticTechnique):
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
    name: str = 'prediction'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methos """

    def _get_gini_predictions(self, recipe):
        """Makes predictions from fitted model with gini method.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predictions from fitted model on test data.

        """
        if hasattr(recipe.model.algorithm, 'predict'):
            return recipe.model.algorithm.predict(
                getattr(recipe.ingredients, 'x_' + self.data_to_review))
        else:
            if self.verbose:
                print('predict method does not exist for',
                      recipe.model.technique.name)
            return None

    def _get_shap_predictions(self, recipe):
        """Makes predictions from fitted model based upon shap values.

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
            'gini' : self._get_gini_predictions,
            'shap' : self._get_shap_predictions}
        return self
