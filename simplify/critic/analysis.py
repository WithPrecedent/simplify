"""
critic.py is the primary control file for evaluating and visualizing the data
and data analysis portions of the siMpLify package. It contains the Analysis
class, which the review and evaluation process.
"""
from dataclasses import dataclass

from simplify.core.base import SimpleClass
from simplify.critic.steps import Summarize, Evaluate, Visualize


@dataclass
class Analysis(SimpleClass):
    """Summarizes, evaluates, and creates visualizations for data and data
    analysis from the siMpLify Harvest and Cookbook.

    Parameters:
        ingredients: an instance of Ingredients (or a subclass).
        recipes: an instance of Recipe or a list of instances of Recipes.
        steps: an ordered list of step names to be completed. This argument
            should only be passed if the user whiches to override the steps
            listed in 'idea.configuration'. The 'evaluate' step can only be
            completed if 'recipes' is passed and much of the 'visualize'
            step cannot be completed without 'recipes'.
        name: a string designating the name of the class which should be
            identical to the section of the idea configuration with relevant
            settings.
        auto_finalize: a boolean value that sets whether the finalize method is
            automatically called when the class is instanced.
        auto_produce: sets whether to automatically call the 'produce' method
            when the class is instanced.
    """
    ingredients : object = None
    recipes : object = None
    steps : object = None
    name : str = 'analysis'
    auto_finalize : bool = True
    auto_produce : bool = False

    def __post_init__(self):
        super().__post_init__()
        return self
    
    """ Private Methods """
    
    def _check_best(self, recipe):
        """Checks if the current recipe is better than the current best recipe
        based upon the primary scoring metric.

        Parameters:
            recipe: an instance of Recipe to be tested versus the current best
                recipe stored in the 'best_recipe' attribute.
        """
        if not hasattr(self, 'best_recipe') or self.best_recipe is None:
            self.best_recipe = recipe
            self.best_recipe_score = self.analysis.review.report.loc[
                    self.analysis.review.report.index[-1],
                    self.listify(self.metrics)[0]]
        elif (self.analysis.review.report.loc[
                self.analysis.review.report.index[-1],
                self.listify(self.metrics)[0]] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.analysis.review.report.loc[
                    self.analysis.review.report.index[-1],
                    self.listify(self.metrics)[0]]
        return self
    
    """ Public Methods """
    
    def draft(self):
        """Sets default options for the crtic's review."""
        self.options = {'summarize' : Summarize,
                        'evaluate' : Evaluate}
        self.checks = ['steps']
        # Locks 'step' attribute at 'critic' for conform methods in package.
        self.step = 'critic'
        return self

    def finalize(self):
        self.reviews = []
        for step in self.steps:
            self.reviews.append(self.options[step])
        return self

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  self.listify(self.metrics)[0], 'metric with a score of',
                  f'{self.best_recipe_score : 4.4f}', 'is:')
            for technique in self.best_recipe.techniques:
                print(technique.capitalize(), ':',
                      getattr(self.best_recipe, technique).technique)
        return

    def produce(self, recipe = None):
        """Evaluates recipe with various tools and finalizes report."""
        if self.verbose:
            print('Evaluating recipes')
        for step in self.steps:
            self.options[step]()
        return self
