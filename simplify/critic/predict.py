"""
.. module:: predict
:synopsis: creates predictions from machine learning models
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass


from simplify.core.iterable import SimpleIterable


@dataclass
class Predict(SimpleIterable):
    """Creates predictions from fitted models for out-of-sample data.

    Args:
        steps(dict(str: SimpleTechnique)): names and related SimpleTechnique classes for
            explaining data analysis models.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_implement (bool): whether to call the 'implement' method when the class
            is instanced.
    """
    steps: object = None
    name: str = 'predict'
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
                'outcomes': ['simplify.critic.steps.predictors', 
                             'PredictOutcomes'],
                'probabilities': ['simplify.critic.steps.predictors', 
                                  'PredictProbabilities'],
                'log_probabilities': ['simplify.critic.steps.predictors', 
                                      'PredictLogProbabilities'],
                'shap': ['simplify.critic.steps.predictors', 
                         'PredictShapProbabilities']}
        return self