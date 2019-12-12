
"""
.. module:: reports
:synopsis: reports for model performance
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

from simplify.critic.collection import CriticTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleDirector
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleDirector subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'classification': ['sklearn.metrics', 'classification_report'],
    'confusion': ['sklearn.metrics', 'confusion_matrix']}


@dataclass
class Reports(CriticTechnique):
    """Creates reports for evaluating models.

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
    name: str = 'reports'
    auto_draft: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def __str__(self):
        # """Prints to console basic results separate from report."""
        # print('These are the results using the', recipe.model.step,
        #       'model')
        # if recipe.splicer.step != 'none':
        # print('Confusion Matrix:')
        # print(self.confusion)
        # print('Classification Report:')
        # print(self.classification_report)
        return self

    def draft(self) -> None:
        super().publish()
        self._options = ManuscriptOptions(options = {
            'classification': ('sklearn.metrics', 'classification_report'),
            'confusion': ('sklearn.metrics', 'confusion_matrix')}
        return self

    def publish(self, recipe):
        self.runtime_parameters = {
            'y_true': getattr(recipe.ingredients, 'y_' + self.data_to_review),
            'y_pred': recipe.predictions}
        super().implement()
        return self

