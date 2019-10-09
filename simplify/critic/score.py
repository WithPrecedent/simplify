"""
.. module:: score
:synopsis: records metrics for model performance
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd
from sklearn import metrics

from simplify.core.iterable import SimpleIterable
from simplify.core.technique import SimpleTechnique


@dataclass
class Score(SimpleIterable):
    """Scores models and prepares reports based upon model type.

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
    name: str = 'score'
    auto_publish: bool = True
    auto_implement: bool = False

    def __post_init__(self):
        self.idea_sections = ['critic']
        super().__post_init__()
        return self

    """ Private Methods """

    def _implement_metrics(self, recipe):
        for column in self.metrics_to_use:
            if column in self.metrics.options:
                if column in self.metrics.prob_options:
                    params = {
                        'y_true': recipe.ingredients.y_test,
                        'y_prob': self.predicted_probs[:, 1]}
                elif column in self.metrics.score_options:
                    params = {
                        'y_true': recipe.ingredients.y_test,
                        'y_score': self.predicted_probs[:, 1]}
                else:
                    params = {
                        'y_true': recipe.ingredients.y_test,
                        'y_pred': self.predictions}
                if column in self.metrics.special_metrics:
                    params.update({column: self.special_metrics[column]})
                result = self.metrics.metrics.options[column](**params)
                if column in self.metrics.negative_metrics:
                    result = -1 * result
                self.report[column] = result 
        return self       
        
    def _set_columns(self):
        self.columns = list(self.options.keys())
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
                'metrics': ['simplify.critic.steps.scorers', 'Metrics'],
                'reports': ['simplify.critic.steps.scorers', 'Reports']}
        self.sequence = ['metrics']
        return self

    def edit(self, name, metric, special_type = None,
             special_parameters = None, negative_metric = False):
        """Allows user to manually add a metric to report."""
        self.options.update({name: metric})
        if special_type in ['probability']:
            self.prob_options.update({name: metric})
        elif special_type in ['scorer']:
            self.score_options.update({name: metric})
        if special_parameters:
           self.special_options.update({name: special_parameters})
        if negative_metric:
           self.negative_options.append[name]
        return self

    def publish(self):
        super().publish()
        self._set_columns()
        self.report = pd.Series(index = self.columns)
        return self

    def implement(self, recipe = None):
        """Prepares the results of a single recipe application to be added to
        the .report dataframe.
        """
        for step in self.sequence: 
            getattr(self, '_implement_' + step)(recipe = recipe)       
        return self