"""
.. module:: explain
:synopsis: explains machine learning results
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.iterables import SimplePlan


@dataclass
class Explain(SimplePlan):
    """Explains model results.

    Args:
        steps(dict(str: SimpleTechnique)): names and related SimpleTechnique classes for
            explaining data analysis models.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
    """

    steps: object = None
    name: str = 'explain'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
                'eli5': ['simplify.critic.steps.explainers', 'Eli5Explain'],
                'shap': ['simplify.critic.steps.explainers', 'ShapExplain'],
                'skater': ['simplify.critic.steps.explainers', 'SkaterExplain']}
        self.return_variables = {
            'eli5': ['feature_importances'],
            'shap': ['feature_importances', 'values', 'interaction_values'],
            'skater': ['feature_importances']}
        return self
