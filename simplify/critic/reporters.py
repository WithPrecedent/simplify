"""
.. module:: reporters
:synopsis: reports for model performance
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd

from simplify.core.base import SimpleSettings
from simplify.critic.critic import Evaluator


@dataclass
class Reporter(SimpleSettings, Evaluator):
    """Base class for report preparation.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']

    """ Core siMpLify Methods """

    def apply(self, data: 'Chapter') -> 'Chapter':
        """Subclasses should provide their own methods."""
        return data


@dataclass
class ConfusionMatrix(Reporter):
    """Summary report for Analyst performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']

    """ Private Methods """

    def _create_report(self,
            actual: Union[np.ndarray, pd.Series],
            prediction: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame(
            self.algorithm(
                actual,
                prediction,
                labels = ['yes', 'no']),
                index = ['actual:yes', 'actual:no'],
                columns =['predicted:yes', 'predicted:no'])

    """ Core siMpLify Methods """

    def apply(self, data: 'Chapter') -> 'Chapter':
        self.algorithm = algorithm.load('algorithm')
        actual = getattr(data.data, '_'.join(
            'y', self.idea['critic']['data_to_review']))
        for key, prediction in data.predictions:
            new_key = '_'.join('classification', key)
            data.reports[new_key] = self._create_report(
                prediction = prediction,
                actual = actual)
        return data


@dataclass
class ClassificationReport(Reporter):
    """Summary report for Analyst performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']

    """ Private Methods """

    def _create_report(self,
            actual: Union[np.ndarray, pd.Series],
            prediction: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame(
            self.algorithm(actual, prediction, output_dict = True)).transpose()

    """ Core siMpLify Methods """

    def apply(self, data: 'Chapter') -> 'Chapter':
        self.algorithm = algorithm.load('algorithm')
        actual = getattr(data.data, '_'.join(
            'y', self.idea['critic']['data_to_review']))
        for key, prediction in data.predictions:
            new_key = '_'.join('classification', key)
            data.reports[new_key] = self._create_report(
                prediction = prediction,
                actual = actual)
        return data


@dataclass
class SimplifyReporter(Reporter):
    """Summary report for Analyst performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']

# @dataclass
# class Article(object):

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     """ Private Methods """

#     def _add_row(self, recipe, report):
#         new_row = pd.Series(index = self.columns)
#         for column, variable in self.required_columns.items():
#             new_row[column] = getattr(recipe, variable)
#         for column in report:
#             new_row[column] = report[column]
#         self.text.loc[len(self.text)] = new_row
#         return self

#     def _check_best(self, recipe):
#         """Checks if the current recipe is better than the current best recipe
#         based upon the primary scoring metric.

#         Args:
#             recipe: an instance of Recipe to be tested versus the current best
#                 recipe stored in the 'best_recipe' attribute.
#         """
#         if not self._exists('best_recipe'):
#             self.best_recipe = recipe
#             self.best_recipe_score = self.article.loc[
#                     self.article.index[-1],
#                     listify(self.metrics)[0]]
#         elif (self.article.loc[
#                 self.article.index[-1],
#                 listify(self.metrics)[0]] > self.best_recipe_score):
#             self.best_recipe = recipe
#             self.best_recipe_score = self.article.loc[
#                     self.article.index[-1],
#                     listify(self.metrics)[0]]
#         return self

#     def _format_step(self, attribute):
#         if getattr(self.recipe, attribute).step in ['none', 'all']:
#             step_column = getattr(self.recipe, attribute).step
#         else:
#             step = getattr(self.recipe, attribute).step
#             parameters = getattr(self.recipe, attribute).parameters
#             step_column = f'{step}, parameters = {parameters}'
#         return step_column

#     def _get_step_name(self, step):
#         """Returns appropriate algorithm to the report attribute."""
#         if step.step in ['none', 'all']:
#             return step.step
#         else:
#             return step.algorithm

#     def print_best(self):
#         """Prints output to the console about the best recipe."""
#         if self.verbose:
#             print('The best test recipe, based upon the',
#                   listify(self.metrics)[0], 'metric with a score of',
#                   f'{self.best_recipe_score: 4.4f}', 'is:')
#             for step in getattr(self,
#                     self.iterator).best_recipe.steps:
#                 print(step.capitalize(), ':',
#                       getattr(getattr(self, self.iterator).best_recipe,
#                               step).step)
#         return

#     def _set_columns(self, recipe):
#         self.required_columns = {
#             'recipe_number': 'number',
#             'options': 'steps',
#             'seed': 'seed',
#             'validation_set': 'using_val_set'}
#         self.columns = list(self.required_columns.keys())
#         self.columns.extend(recipe.steps)
#         for step in self.steps:
#             if (hasattr(getattr(self, step), 'columns')
#                     and getattr(self, step).name != 'summarize'):
#                 self.columns.extend(getattr(self, step).columns)
#         return self

#     def _start_report(self, recipe):
#         self._set_columns(recipe = recipe)
#         self.text = pd.DataFrame(columns = self.columns)
#         return self

#     """ Public Import/Export Methods """

#     def save(self, report = None):
#         """Exports the review report to disk.

#         Args:
#             review(Review.report): 'report' from an instance of review
#         """
#         self.filer.save(
#             variable = report,
#             folder = self.filer.experiment,
#             file_name = self.model_type + '_review',
#             file_format = 'csv',
#             header = True)
#         return