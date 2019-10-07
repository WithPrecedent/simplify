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
        super().__post_init__()
        return self

    """ Private Methods """

    def _check_best(self, recipe):
        """Checks if the current recipe is better than the current best recipe
        based upon the primary scoring metric.

        Args:
            recipe: an instance of Recipe to be tested versus the current best
                recipe stored in the 'best_recipe' attribute.
        """
        if not self.exists('best_recipe'):
            self.best_recipe = recipe
            self.best_recipe_score = self.report.loc[
                    self.report.index[-1],
                    self.listify(self.metrics)[0]]
        elif (self.report.loc[
                self.report.index[-1],
                self.listify(self.metrics)[0]] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.report.loc[
                    self.report.index[-1],
                    self.listify(self.metrics)[0]]
        return self

    def _set_columns(self):
        self.columns = list(self.options.keys())
        return self

    """ Public Tool Methods """

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  self.listify(self.metrics)[0], 'metric with a score of',
                  f'{self.best_recipe_score: 4.4f}', 'is:')
            for technique in getattr(self,
                    self.iterable).best_recipe.techniques:
                print(technique.capitalize(), ':',
                      getattr(getattr(self, self.iterable).best_recipe,
                              technique).technique)
        return

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
                'metrics': ['simplify.critic.steps.scorers', 'Metrics'],
                'reports': ['simplify.critic.steps.scorers', 'Reports']}
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
        self._set_columns()
        return self

    def implement(self, ingredients = None, recipes = None):
        """Prepares the results of a single recipe application to be added to
        the .report dataframe.
        """
        scores = pd.Series(index = self.columns)
        for column, value in self.options.items():
            if column in self.metrics:
                if column in self.prob_options:
                    params = {'y_true': self.recipe.ingredients.y_test,
                              'y_prob': self.predicted_probs[:, 1]}
                elif column in self.score_options:
                    params = {'y_true': self.recipe.ingredients.y_test,
                              'y_score': self.predicted_probs[:, 1]}
                else:
                    params = {'y_true': self.recipe.ingredients.y_test,
                              'y_pred': self.predictions}
                if column in self.special_metrics:
                    params.update({column: self.special_metrics[column]})
                result = value(**params)
                if column in self.negative_metrics:
                    result = -1 * result
                scores[column] = result
        return scores