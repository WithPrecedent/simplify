"""
critic.py is the primary control file for evaluating the modeling of data.

Contents:
    Analysis: primary class constructing and controlling model evaluation.
    Review: class for accumulating metrics and evaluations of the data and
        model.
"""
from dataclasses import dataclass

from simplify.core.base import SimpleManager, SimplePlan
from simplify.critic.steps.summarize import Summarize
from simplify.critic.steps.evaluate import Evaluate
from simplify.critic.steps.score import Score
from simplify.critic.steps.report import Report


@dataclass
class Analysis(SimpleManager):
    """Summarizes, evaluates, and creates visualizations for data and data
    analysis from the siMpLify Harvest and Cookbook.

    Args:
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

    """ Public Methods """

    def draft(self):
        """Sets default options for the Critic's analysis."""
        self.options = {'summarizer' : Summarize,
                        'evaluator' : Evaluate,
                        'scorer' : Score,
                        'reporter' : Report}
        self.checks = ['steps']
        # Locks 'step' attribute at 'critic' for conform methods in package.
        self.step = 'critic'
        # Sets 'manager_type' so that proper parent methods are used.
        self.manager_type = 'serial'
        # Sets 'plan_class' to allow use of parent methods.
        self.plan_class = Review
        self.plan_iterable = 'review'
        return self

    def finalize(self):
        """Calls the appropriate finalize method based upon 'manager_type' of
        class.
        """
        super().finalize()
        return self

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  self.listify(self.metrics)[0], 'metric with a score of',
                  f'{self.best_recipe_score : 4.4f}', 'is:')
            for technique in getattr(self,
                    self.plan_iterable).best_recipe.techniques:
                print(technique.capitalize(), ':',
                      getattr(getattr(self, self.plan_iterable).best_recipe,
                              technique).technique)
        return

    def produce(self, recipes = None):
        """Evaluates recipe with various tools and finalizes report."""
        for recipe in self.listify(recipes):
            if self.verbose:
                print('Evaluating', recipe.name + 's')
            for step, technique in getattr(self, self.plan_iterable).items():
                technique.produce(recipe = recipe)
        return self


@dataclass
class Review(SimplePlan):
    """Stores machine learning experiment results.

    Report creates and stores a results report and other general
    scorers/metrics for machine learning based upon the type of model used in
    the siMpLify package. Users can manually add metrics not already included
    in the metrics dictionary by passing them to Results.edit_metric.

    Attributes:
        name: a string designating the name of the class which should be
            identical to the section of the idea with relevant settings.
        auto_finalize: sets whether to automatically call the finalize method
            when the class is instanced. If you do not draft to make any
            adjustments to the options or metrics beyond the idea, this option
            should be set to True. If you draft to make such changes, finalize
            should be called when those changes are complete.
    """

    techniques : object = None
    name : str = 'review'
    auto_finalize: bool = True

    def __post_init__(self):
        self.idea_sections = ['analysis']
        super().__post_init__()
        return self

    def _check_best(self, recipe):
        """Checks if the current recipe is better than the current best recipe
        based upon the primary scoring metric.

        Args:
            recipe: an instance of Recipe to be tested versus the current best
                recipe stored in the 'best_recipe' attribute.
        """
        if not hasattr(self, 'best_recipe') or self.best_recipe is None:
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

    def produce(self, recipes):
        for recipe in self.listify(recipes):
            self._check_best()
            for step, technique in self.techniques.items():
                recipe = technique.produce(recipe = recipe)
        return self