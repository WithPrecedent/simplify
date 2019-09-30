"""
.. module:: analysis
  :synopsis: core classes for Critic subpackage.
  :author: Corey Rayburn Yung
  :copyright: 2019
  :license: CC-BY-NC-4.0

This is the primary control file for evaluating, summarizing, and analyzing
data, as well as machine learning and other statistical models.

Contents:
    Analysis: primary class for model evaluation and preparing reports about
        that evaluation and data.
    Review: class for storing metrics, evaluations, and reports related to data
        and models.
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.base import SimpleManager, SimplePlan


@dataclass
class Analysis(SimpleManager):
    """Summarizes, evaluates, and creates visualizations for data and data
    analysis from the siMpLify Harvest and Cookbook.

    Args:
        steps: an ordered list of step names to be completed. This argument
            should only be passed if the user whiches to override the steps
            listed in 'idea.configuration'.
        name (str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_finalize (bool): whether to call the 'finalize' method when the
            class is instanced.
        auto_produce (bool): whether to call the 'produce' method when the class
            is instanced.
    """
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

    def _finalize_report(self):
        self._set_columns()
        self.report = pd.DataFrame(columns = self.columns)
        return self

    def _format_step(self, attribute):
        if getattr(self.recipe, attribute).technique in ['none', 'all']:
            step_column = getattr(self.recipe, attribute).technique
        else:
            technique = getattr(self.recipe, attribute).technique
            parameters = getattr(self.recipe, attribute).parameters
            step_column = f'{technique}, parameters = {parameters}'
        return step_column

    def _set_columns(self):
        self.columns = list(self.columns_map.keys())
        for number, instance in getattr(self, self.plan_iterable).items():
            if hasattr(instance, 'columns'):
                self.columns.extend(instance.columns)
        return self

    """ Public Tool Methods """

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

    """ Core siMpLify methods """

    def draft(self):
        """Sets default options for the Critic's analysis."""
        super().draft()
        self.options = {
                'summarize': ['simplify.critic.summarize', 'Summarize'],
                'explain': ['simplify.critic.explain', 'Explain'],
                'rank': ['simplify.critic.rank', 'Rank'],
                'predict': ['simplify.critic.predict', 'Predict'],
                'score': ['simplify.critic.score', 'Score']}
        # Locks 'step' attribute at 'critic' for conform methods in package.
        self.step = 'critic'
        # Sets 'manager_type' so that proper parent methods are used.
        self.manager_type = 'serial'
        # Sets plan-related attributes to allow use of parent methods.
        self.plan_iterable = 'reviews'
        self.columns_map = {'recipe_number' : 'number',
                            'options' : 'techniques',
                            'seed' : 'seed',
                            'validation_set' : 'val_set'}
        return self

    def finalize(self):
        super().finalize()
        self._finalize_report()
        return self

    def produce(self, ingredients = None, recipes = None):
        """Evaluates recipe with various tools and finalizes report.

        Args:
            ingredients (Ingredients): an instance or subclass instance of
                Ingredients.
            recipes (list or Recipe): a Recipe or a list of Recipes.
        """
        for recipe in self.listify(recipes):
            if self.verbose:
                print('Testing', recipe.name, str(recipe.number))
            self._check_best(recipe = recipe)
            row = pd.Series(index = self.columns)
            for column, value in self.columns.items():
                if isinstance(getattr(recipe, value), object):
                    row[column] = self._format_step(value)
                else:
                    row[column] = getattr(self.recipe, value)
            self.report.loc[len(self.report)] = row
        return self


@dataclass
class Review(SimplePlan):
    """Stores machine learning experiment results.

    Review creates and stores a results report and other general
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

    steps : object = None
    number : int = 0
    name : str = 'review'
    auto_finalize: bool = True
    auto_produce : bool = False

    def __post_init__(self):
        self.idea_sections = ['analysis']
        super().__post_init__()
        return self

    def _set_columns(self):
        """Sets columns and options for report."""
        self.columns = {'recipe_number' : 'number',
                        'options' : 'techniques',
                        'seed' : 'seed',
                        'validation_set' : 'val_set'}

        return self


    def _check_technique_name(self, step):
        """Returns appropriate algorithm to the report attribute."""
        if step.technique in ['none', 'all']:
            return step.technique
        else:
            return step.algorithm

    def _classifier_report(self):
        self.classifier_report_default = metrics.classification_report(
                self.recipe.ingredients.y_test,
                self.predictions)
        self.classifier_report_dict = metrics.classification_report(
                self.recipe.ingredients.y_test,
                self.predictions,
                output_dict = True)
        self.classifier_report = pd.DataFrame(
                self.classifier_report_dict).transpose()
        return self

    def _confusion_matrix(self):
        self.confusion = metrics.confusion_matrix(
                self.recipe.ingredients.y_test, self.predictions)
        return self

    def _cluster_report(self):
        return self


    def _regressor_report(self):
        return self



    def _print_classifier_results(self, recipe):
        """Prints to console basic results separate from report."""
        print('These are the results using the', recipe.model.technique,
              'model')
        if recipe.splicer.technique != 'none':
            print('Testing', recipe.splicer.technique, 'predictors')
        print('Confusion Matrix:')
        print(self.confusion)
        print('Classification Report:')
        print(self.classification_report)
        return self


    def draft(self):
        self.data_variable = 'recipes'
        return self

    def produce(self, recipes):
        setattr(self, self.data_variable, self.listify(recipes))
        for recipe in getattr(self, self.data_variable):
            self._check_best(recipe = recipe)
            for step, technique in self.techniques.items():
                technique.produce(recipe = recipe)
        return self