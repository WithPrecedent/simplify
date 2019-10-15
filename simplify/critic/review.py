"""
.. module:: review
:synopsis: model critic
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.decorators import numpy_shield
from simplify.core.base import SimpleClass
from simplify.core.iterable import SimpleIterable
from simplify.core.technique import SimpleTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'summary': ['simplify.critic.steps.summarize', 'Summarize'],
    'explanation': ['simplify.critic.steps.explain', 'Explain'],
    'prediction': ['simplify.critic.steps.predict', 'Predict'],
    'probabilities': ['simplify.critic.steps.probability', 'Probability'],
    'ranking': ['simplify.critic.steps.rank', 'Rank'],
    'metrics': ['simplify.critic.steps.metrics', 'Metrics'],
    'test': ['simplify.critic.steps.test', 'Test'],
    'report': ['simplify.critic.steps.report', 'Report']}


@dataclass
class Review(SimpleIterable):
    """Builds tools for evaluating, explaining, and creating predictions from
    data and machine learning models.

    Args:
        steps(dict(str: CriticTechnique)): names and related CriticTechnique
            classes for analyzing fitted models.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced.
        auto_implement(bool): whether to call the 'implement' method when the
            class is instanced.

    Since this class is a subclass to SimpleIterable and SimpleClass, all
    documentation for those classes applies as well.

    """
    steps: object = None
    name: str = 'critic'
    auto_publish: bool = True
    auto_implement: bool = False

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Core siMpLify methods """

    def draft(self):
        """Sets default options for the Critic's analysis."""
        super().draft()
        # Locks 'step' attribute at 'critic' for conform methods in package.
        self.depot.step = 'critic'
        return self

    def publish(self):
        Narrative.options = self.options
        Narrative.sequence = self.sequence
        super().publish()
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
            for step in self.sequence:
                getattr(self, step).implement(recipe = recipe)
                self._infuse_attributes(instance = getattr(self, step))
                if step in ['score']:
                    print('score_report', self.score.report)
                    self._add_row(recipe = recipe, report = self.score.report)
                    self.check_best()
        return self


@dataclass
class Narrative(SimpleIterable):

    number: int = 0
    steps: object = None
    name: str = 'narrative'
    auto_publish: bool = True

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.sequence_setting = 'critic_steps'
        return self

    def publish(self):
        super().publish()
        return self

    def implement(self, recipe):
        """Applies the recipe steps to the passed ingredients."""
        for step in self.sequence:
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
        print('report', report)
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
            'options': 'sequence',
            'seed': 'seed',
            'validation_set': 'using_val_set'}
        self.columns = list(self.required_columns.keys())
        self.columns.extend(recipe.sequence)
        for step in self.sequence:
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


@dataclass
class CriticTechnique(SimpleTechnique):
    """Parent Class for techniques in the Critic package.

    This subclass of SimpleTechnique differs from other SimpleTechniques
    because the parameters and algorithm are not joined until the 'implement'
    stage. This is due to the algorithm needed information from the passed
    'recipe' before the algorithm is called. And the techniques ordinarily do
    not have scikit-learn compatible 'fit', 'transform', and 'fit_transform'
    methods.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'generic_critic_technique'
    auto_publish: bool = True

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    """ Core siMpLify Public Methods """

    def publish(self):
        """Finalizes settings.."""
        # Runs attribute checks from list in 'checks' attribute (if it exists).
        self._run_checks()
        # Converts values in 'options' to classes by lazily importing them.
        self._lazily_import()
        return self

    def implement(self, recipe, **kwargs):
        """Returns recipe with feature importances added.

        Args:
            recipe(Recipe): an instance of Recipe or a subclass.

        """
        if self.technique != 'none':
            if not hasattr(self, 'no_parameters') and not self.no_parameters:
                self._set_parameters()
            setattr(recipe, self.technique + '_' + self.name,
                    self.options[self.technique](recipe = recipe))
            if not hasattr(recipe, self.name):
                setattr(recipe, self.name, {})
            setattr(recipe, self.name).update(
                    {self.name: getattr(
                    self, self.technique + '_' + self.name)})
        return recipe