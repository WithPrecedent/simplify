"""
critic.py is the primary control file for evaluating and visualizing the data
and data analysis portions of the siMpLify package. It contains the Analysis
class, which the review and evaluation process.
"""
from dataclasses import dataclass

import pandas as pd

from simplify.core.base import SimpleClass
from simplify.core.technique import Technique
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
            listed in 'menu.configuration'. The 'evaluate' step can only be
            completed if 'recipes' is passed and much of the 'visualize'
            step cannot be completed without 'recipes'.
        name: a string designating the name of the class which should be
            identical to the section of the menu configuration with relevant
            settings.
        auto_prepare: a boolean value that sets whether the prepare method is
            automatically called when the class is instanced.
        auto_perform: sets whether to automatically call the 'perform' method
            when the class is instanced.
    """
    ingredients : object = None
    recipes : object = None
    steps : object = None
    name : str = 'analysis'
    auto_prepare : bool = True
    auto_perform : bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    def plan(self):
        """Sets default options for the Analysis review."""
        self.options = {'summarize' : Summarize,
                        'evaluate' : Evaluate,
                        'visualize' : Visualize}
        self.checks = ['steps']
        self.step = 'review'
        self.export_folder = 'experiment'
        return self

    def prepare(self):

        self.reviews = []
        for step in self.steps:
            self.reviews.append(self.options[step])
        return self

    def perform(self, recipe = None):
        """Evaluates recipe with various tools and prepares report."""
        if self.verbose:
            print('Evaluating recipes')
        for step in self.steps:
            self.options[step]()
        return self
