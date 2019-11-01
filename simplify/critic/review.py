"""
.. module:: review
:synopsis: model critic
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from simplify.core.decorators import numpy_shield
from simplify.core.base import SimpleClass
from simplify.core.package import SimplePackage
from simplify.core.plans import SimplePlan
from simplify.core.technique import CriticTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'summary': ['simplify.critic.techniques.summarize', 'Summarize'],
    'explanation': ['simplify.critic.techniques.explain', 'Explain'],
    'prediction': ['simplify.critic.techniques.predict', 'Predict'],
    'probabilities': ['simplify.critic.techniques.probability', 'Probability'],
    'ranking': ['simplify.critic.techniques.rank', 'Rank'],
    'metrics': ['simplify.critic.techniques.metrics', 'Metrics'],
    'test': ['simplify.critic.techniques.test', 'Test'],
    'report': ['simplify.critic.techniques.report', 'Report']}


@dataclass
class Review(SimplePackage):
    """Builds tools for evaluating, explaining, and creating predictions from
    data and machine learning models.

    Args:
        techniques(dict(str: CriticTechnique)): names and related CriticTechnique
            classes for analyzing fitted models.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class.
        auto_draft(bool): whether to call the 'publish' method when the
            class is instanced.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced.

    Since this class is a subclass to SimplePackage and SimpleClass, all
    documentation for those classes applies as well.

    """
    techniques: object = None
    name: str = 'critic'
    auto_draft: bool = True
    auto_publish: bool = False
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Core siMpLify methods """

    def draft(self):
        """Sets default options for the Critic's analysis."""
        # Sets comparer class for storing parallel plans
        self.comparer = Narrative
        self.comparer_iterable = 'narratives'
        # Locks 'step' attribute at 'critic' for state dependent methods.
        self.depot.step = 'critic'
        super().draft()
        return self

    def implement(self, recipes = None):
        """Evaluates recipe with various tools and publishs report.

        Args:
            recipes(dict(str: Recipe) or Recipe): a Recipe or a dict of Recipes.
                The recipes included should have fit models for this class's
                methods to work.
        """
        if not isinstance(recipes, dict):
            recipes = {recipes.number: recipes}
        self.recipes = recipes
        # Initializes comparative model report with set columns.
        if not self.exists('article'):
            self.article = Article()
        # Iterates through 'recipes' to gather review information.
        for number, recipe in self.recipes.items():
            if self.verbose:
                print('Reviewing', recipe.name, str(number))
            step_reviews = {}
            for step in self.order:
                getattr(self, step).implement(recipe = recipe)
                self._infuse_return_variables(instance = getattr(self, step))
                if step in ['score']:
                    print('score_report', self.score.report)
                    self._add_row(recipe = recipe, report = self.score.report)
                    self.check_best()
        return self


@dataclass
class Narrative(SimplePlan):

    number: int = 0
    techniques: object = None
    name: str = 'narrative'
    auto_draft: bool = True

    def __post_init__(self):
        self.idea_sections = ['critic']
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        if not self.options:
            self.options = DEFAULT_OPTIONS
        self.order_setting = 'critic_techniques'
        self.is_comparer = True
        return self

    def implement(self, recipe):
        """Applies the recipe techniques to the passed ingredients."""
        for step in self.order:
            if step in ['summary']:
                pass
            elif step in ['prediction', 'probabilities', 'explanation']:
                getattr(self, step).implement(
                       recipe = recipe)
            elif step in ['ranking', 'score']:
                getattr(self, step).implement(
                       recipe = recipe,
                       prediction = self.prediction)
            if self.export_results:
                pass
        return self


@dataclass
class Article(SimpleClass):

    def __post_init__(self):
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
        if not self.exists('best_recipe'):
            self.best_recipe = recipe
            self.best_recipe_score = self.article.loc[
                    self.article.index[-1],
                    self.listify(self.metrics)[0]]
        elif (self.article.loc[
                self.article.index[-1],
                self.listify(self.metrics)[0]] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.article.loc[
                    self.article.index[-1],
                    self.listify(self.metrics)[0]]
        return self

    def _format_step(self, attribute):
        if getattr(self.recipe, attribute).technique in ['none', 'all']:
            step_column = getattr(self.recipe, attribute).technique
        else:
            technique = getattr(self.recipe, attribute).technique
            parameters = getattr(self.recipe, attribute).parameters
            step_column = f'{technique}, parameters = {parameters}'
        return step_column

    def _get_technique_name(self, step):
        """Returns appropriate algorithm to the report attribute."""
        if step.technique in ['none', 'all']:
            return step.technique
        else:
            return step.algorithm

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  self.listify(self.metrics)[0], 'metric with a score of',
                  f'{self.best_recipe_score: 4.4f}', 'is:')
            for technique in getattr(self,
                    self.iterator).best_recipe.techniques:
                print(technique.capitalize(), ':',
                      getattr(getattr(self, self.iterator).best_recipe,
                              technique).technique)
        return

    def _set_columns(self, recipe):
        self.required_columns = {
            'recipe_number': 'number',
            'options': 'order',
            'seed': 'seed',
            'validation_set': 'using_val_set'}
        self.columns = list(self.required_columns.keys())
        self.columns.extend(recipe.order)
        for step in self.order:
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
        """Exports the review report to disc.

        Args:
            review(Review.report): 'report' from an instance of review
        """
        self.depot.save(
            variable = report,
            folder = self.depot.experiment,
            file_name = self.model_type + '_review',
            file_format = 'csv',
            header = True)
        return

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        return self

    def publish(self):
        super().publish()
        return self