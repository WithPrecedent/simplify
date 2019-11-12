"""
.. module:: siMpLify project
:synopsis: entry point for implementing siMpLify subpackages
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from simplify import timer
from simplify.core.depot import Depot
from simplify.core.idea import Idea
from simplify.core.ingredients import Ingredients


@timer('simplify project')
@dataclass
class Project(object):
    """Controller class for siMpLify projects.

    Args:
        idea (Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located. Once an
            Idea instance is createds, it is automatically an attribute to all
            other SimpleComposite subclasses that are instanced in the future.
            Required.
        depot (Depot or str): an instance of Depot or a string containing the
            full path of where the root folder should be located for file
            output. A Depot instance contains all file path and import/export
            methods for use throughout the siMpLify package. Once a Depot
            instance is created, it is automatically an attribute of all other
            SimpleComposite subclasses that are instanced in the future. Default
            is None.
        ingredients (Ingredients, DataFrame, Series, ndarray, or str): an
            instance of Ingredients, a string containing the full file path of
            where a data file for a pandas DataFrame or Series is located, a
            string containing a,file name in the default data folder, as defined
            in the shared Depot instance, a DataFrame, a Series, or numpy
            ndarray. If a DataFrame, ndarray, or string is provided, the
            resultant DataFrame is stored at the 'df' attribute in a new
            Ingredients instance. Default is None
        steps (List[str] or str): names of techniques to be applied. These
            names should match keys in the 'options' attribute. If using the
            Idea instance settings, this argument should not be passed. Default
            is None.
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Default is 'simple_package', but should be overwritten to
            match settings in the Idea instance.

    """
    idea: Union[Idea, str]
    depot: Optional[Union[Depot, str]] = None
    ingredients: Optional[Union[Ingredients, pd.DataFrame, pd.Series,
                                np.ndarray, str]] = None
    steps: Optional[Union[List[str], str]] = None
    name: Optional[str] = 'simplify'

    def __post_init__(self) -> None:
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        self.draft()
        return self

    """ Private Methods """

    def _draft_steps(self) -> None:
        for step, planner in self.steps.items():
            if self.idea['general']['verbose']:
                print('Initializing', step)
            setattr(self, step, planner())
            getattr(self, step).research(distributors = [self.idea, self.depot])
            getattr(self, step).draft()
        return self

    def _get_parameters(self, step: str) -> Dict[str, 'SimplePlanner']:
        parameters = []
        for parameter in self.publish_paramters[step]:
            parameters.append(getattr(self, parameter))
        return parameters

    def _set_steps(self) -> None:
        """Creates 'steps' containing technique builder classes."""
        if self.steps is None:
            self.steps = self.idea['simplify']['simplify_steps']
        new_steps = {}
        for step in self.steps:
            try:
                step_class = getattr(import_module(
                    self.options[step][0]),
                        self.options[step][1])
                new_steps[step] = step_class
            except KeyError:
                error = ' '.join([step,
                                  'does not match an option in', self.name])
                raise KeyError(error)
        self.steps = new_steps
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        # Sets step options with information for module importation.
        self.options = {
            'farmer': ('simplify.farmer', 'Almanac'),
            'chef': ('simplify.chef', 'Cookbook'),
            'actuary': ('simplify.actuary', 'Ledger'),
            'critic': ('simplify.critic', 'Review'),
            'artist': ('simplify.artist', 'Canvas')}
        # Sets parameters to be sent to each step's publish method.
        self.publish_parameters = {
            'farmer': (),
            'chef': ('ingredients'),
            'actuary': ('chef.ingredients'),
            'critic': ('chef.ingredients', 'chef.recipes'),
            'artist': ('critic.ingredients', 'chef.recipes', 'critic.reviews')}
        # Completes an Idea instance.
        self.idea = Idea(configuration = self.idea)
        # Completes a Depot instance.
        self.depot = Depot(root_folder = self.depot)
        # Completes an Ingredients instance.
        self.ingredients = Ingredients(df = self.ingredients)
        # Finalizes 'steps' attribute.
        self._set_steps()
        self._draft_steps()
        return self

    def publish(self) -> None:
        """Applies 'steps' to 'ingredients'."""
        for step in self.steps.keys():
            parameters = self._get_parameters(step = step)
            setattr(self, step, getattr(self, step).publish(*parameters))
        return self