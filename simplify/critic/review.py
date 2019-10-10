"""
.. module:: review
:synopsis: core classes for Critic subpackage.
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.iterable import SimpleIterable


@dataclass
class Review(SimpleIterable):
    """Builds tools for evaluating, explaining, and creating predictions from
    data and machine learning models.

    Args:
        ingredients(Ingredients or str): an instance of Ingredients of a string
            containing the full file path of where a supported file type that
            can be loaded into a pandas DataFrame is located. If it is a string,
            the loaded DataFrame will be bound to a new ingredients instance as
            the 'df' attribute.
        steps(dict(str: SimpleIterable)): names and related SimpleIterable
            classes for analyzing fitted models.
        recipes(Recipe or list(Recipe)): a list or single Recipe to be reviewed.
            This argument need not be passed when the class is instanced. It
            can be passed directly to the 'implement' method as well.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced.
        auto_implement(bool): whether to call the 'implement' method when the
            class is instanced.

    Since this class is a subclass to SimpleIterable and SimpleClass, all
    documentation for those classes applies as well.

    """

    ingredients: object = None
    steps: object = None
    recipes: object = None
    name: str = 'critic'
    auto_publish: bool = True
    auto_implement: bool = False

    def __post_init__(self):
        self.idea_sections = ['chef']
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
        self.report.loc[len(self.report)] = new_row
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
        self.report = pd.DataFrame(columns = self.columns)
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

    """ Core siMpLify methods """

    def draft(self):
        """Sets default options for the Critic's analysis."""
        super().draft()
        self.options = {
            'summary': ['simplify.critic.summarize', 'Summarize'],
            'prediction': ['simplify.critic.predict', 'Predict'],
            'probabilities': ['simplify.critic.probability', 'Probability'],
            'explanation': ['simplify.critic.explain', 'Explain'],
            'ranking': ['simplify.critic.rank', 'Rank'],
            'score': ['simplify.critic.score', 'Score']}
        # Locks 'step' attribute at 'critic' for conform methods in package.
        self.depot.step = 'critic'
        self.return_variables = {
            'score' : ['best_recipe', 'score', 'best_recipe']}
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
            recipes = {'1': recipes}
        self.recipes = recipes
        # Initializes comparative model report with set columns.
        if not self.exists('report'):
            self._start_report(recipe = recipes['1'])
        print('columns', self.columns)
        # Iterates through 'recipes' to gather review information.
        for number, recipe in self.recipes.items():
            if self.verbose:
                print('Reviewing', recipe.name, str(recipe.number))
            step_reviews = {}
            for step in self.sequence:
                getattr(self, step).implement(recipe = recipe)
                self._infuse_attributes(instance = getattr(self, step))
                if step in ['score']:
                    print('score_report', self.score.report)
                    self._add_row(recipe = recipe, report = self.score.report)
                    self.check_best()
        self.print_best
        print(self.report)
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
        if not self.options:
            self.options = {
                'prediction': ['simplify.critic.predict', 'Predict'],
                'probabilities': ['simplify.critic.probability', 'Probability'],
                'explanation': ['simplify.critic.explain', 'Explain'],
                'ranking': ['simplify.critic.rank', 'Rank'],
                'score': ['simplify.critic.score', 'Score']}
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
    
        