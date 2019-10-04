"""
.. module:: explain
:synopsis: explains machine learning results
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.plan import SimplePlan
from simplify.core.step import SimpleStep


@dataclass
class Explain(SimplePlan):
    """Explains model results.

    Args:
        steps(dict(str: SimpleStep)): names and related SimpleStep classes for
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
        self.options = {
                'eli5': ['simplify.critic.steps.explianers', 'Eli5Explain'],
                'shap': ['simplify.critic.steps.explianers', 'ShapExplain'],
                'skater': ['simplify.critic.steps.explianers', 'SkaterExplain']}
        return self

    def implement(self, recipe):
        """Creates a dictionary of 'reports' from explainer techniques.

        Args:
            recipe (Recipe): a Recipe with a fitted model.
        """
        for step_name, step_instance in self.options.items():
            if step_name in self.explainers:

        return self
