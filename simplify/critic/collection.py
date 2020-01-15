"""
.. module:: review
:synopsis: model critic
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from simplify.core.typesetter import SimpleDirector
from simplify.core.book import Book
from simplify.core.book import Chapter


@dataclass
class Collection(Book):
    """Builds tools for evaluating, explaining, and creating predictions from
    data and machine learning models.

    Args:


    """
    name: Optional[str] = 'critic'
    steps: Optional[Dict[str, 'SimpleDirector']] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Core siMpLify methods """

    def draft(self) -> None:
        """Sets default options for the Critic's analysis."""
        self._options = SimpleCatalog(options = {
            'explanation': ('simplify.critic.steps.explain', 'Explain'),
            'prediction': ('simplify.critic.steps.predict', 'Predict'),
            'probabilities': ('simplify.critic.steps.probability',
                              'Probability'),
            'ranking': ('simplify.critic.steps.rank', 'Rank'),
            'metrics': ('simplify.critic.steps.metrics', 'Metrics'),
            'reports': ('simplify.critic.steps.reports', 'Reports')}
        # Sets plan container
        self.chapter_type = Review
        return self


@dataclass
class Review(Chapter):

    def __post_init__(self) -> None:
        super().__post_init__()
        return self


@dataclass
class Article(SimpleDirector):

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _add_row(self, recipe, report):
        new_row = pd.Series(index = self.columns)
        for column, variable in self.required_columns.items():
            new_row[column] = getattr(recipe, variable)
        for column in report:
            new_row[column] = report[column]
        self.text.loc[len(self.text)] = new_row
        return self

    def _check_best(self, recipe):
        """Checks if the current recipe is better than the current best recipe
        based upon the primary scoring metric.

        Args:
            recipe: an instance of Recipe to be tested versus the current best
                recipe stored in the 'best_recipe' attribute.
        """
        if not self._exists('best_recipe'):
            self.best_recipe = recipe
            self.best_recipe_score = self.article.loc[
                    self.article.index[-1],
                    listify(self.metrics)[0]]
        elif (self.article.loc[
                self.article.index[-1],
                listify(self.metrics)[0]] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.article.loc[
                    self.article.index[-1],
                    listify(self.metrics)[0]]
        return self

    def _format_step(self, attribute):
        if getattr(self.recipe, attribute).step in ['none', 'all']:
            step_column = getattr(self.recipe, attribute).step
        else:
            step = getattr(self.recipe, attribute).step
            parameters = getattr(self.recipe, attribute).parameters
            step_column = f'{step}, parameters = {parameters}'
        return step_column

    def _get_step_name(self, step):
        """Returns appropriate algorithm to the report attribute."""
        if step.step in ['none', 'all']:
            return step.step
        else:
            return step.algorithm

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  listify(self.metrics)[0], 'metric with a score of',
                  f'{self.best_recipe_score: 4.4f}', 'is:')
            for step in getattr(self,
                    self.iterator).best_recipe.steps:
                print(step.capitalize(), ':',
                      getattr(getattr(self, self.iterator).best_recipe,
                              step).step)
        return

    def _set_columns(self, recipe):
        self.required_columns = {
            'recipe_number': 'number',
            'options': 'steps',
            'seed': 'seed',
            'validation_set': 'using_val_set'}
        self.columns = list(self.required_columns.keys())
        self.columns.extend(recipe.steps)
        for step in self.steps:
            if (hasattr(getattr(self, step), 'columns')
                    and getattr(self, step).name != 'summarize'):
                self.columns.extend(getattr(self, step).columns)
        return self

    def _start_report(self, recipe):
        self._set_columns(recipe = recipe)
        self.text = pd.DataFrame(columns = self.columns)
        return self

    """ Public Import/Export Methods """

    def save(self, report = None):
        """Exports the review report to disk.

        Args:
            review(Review.report): 'report' from an instance of review
        """
        self.inventory.save(
            variable = report,
            folder = self.inventory.experiment,
            file_name = self.model_type + '_review',
            file_format = 'csv',
            header = True)
        return

    """ Core siMpLify Methods """

    def draft(self) -> None:
        super().draft()
        return self

    def publish(self):
        super().publish()
        return self