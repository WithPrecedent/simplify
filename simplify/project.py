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
from typing import Any, Dict, Iterable, List, Union
import warnings

import numpy as np
import pandas as pd

from simplify import timer
from simplify.core.base import SimpleClass
from simplify.core.depot import Depot
from simplify.core.idea import Idea
from simplify.core.ingredients import Ingredients
from simplify.core.planner import SimplePlanner


@timer('siMpLify project')
@dataclass
class Project(SimplePlanner):
    """Controller class for siMpLify projects.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        idea (Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located. Once an
            Idea instance is createds, it is automatically an attribute to all
            other SimpleClass subclasses that are instanced in the future.
        depot (Depot or str): an instance of Depot or a string containing the
            full path of where the root folder should be located for file
            output. A Depot instance contains all file path and import/export
            methods for use throughout the siMpLify package. Once a Depot
            instance is created, it is automatically an attribute of all other
            SimpleClass subclasses that are instanced in the future.
        ingredients (Ingredients, DataFrame, Series, ndarray, or str): an
            instance of Ingredients, a string containing the full file path of
            where a data file for a pandas DataFrame or Series is located, a
            string containing a,file name in the default data folder, as defined
            in the shared Depot instance, a DataFrame, a Series, or numpy
            ndarray. If a DataFrame, ndarray, or string is provided, the
            resultant DataFrame is stored at the 'df' attribute in a new
            Ingredients instance.
        steps (List[str], or str): names of all subpackages to be used from the
            siMpLify package. This argument only needs to be passed if the
            subpackages to be used are different than those listed in 'idea'.
            This can be useful in running tests with a particular subpackage.

    """
    name: str = 'simplify'
    idea: Union[Idea, str] = None
    depot: Union[Depot, str, None] = None
    ingredients: Union[Ingredients, pd.DataFrame, pd.Series, np.ndarray, str,
                       None] = None
    steps: Union[List[str], str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_depot(self) -> None:
        """Completes a Depot instance for the 'depot' attribute.

        If a file path is passed to 'depot', a Depot instance is created with
        that folder as 'root_folder'.

        If 'depot' is None, a Depot instance is created with default options.

        If a completed Depot instance was passed, it is left intact.

        Raises:
            TypeError: if 'depot' is neither a str, None, nor Depot instance.

        """
        if self.depot is None:
            self.depot = Depot()
        elif isinstance(self.depot, str):
            self.depot = Depot(root_folder = self.depot)
        elif isinstance(self.depot, Depot):
            pass
        else:
            error = 'depot must be None, str, or Depot instance'
            raise TypeError(error)
        # Injects base class with 'depot' instance.
        SimpleClass.depot = self.depot
        return self

    def _draft_idea(self) -> None:
        """Completes an Idea instance for the 'idea' attribute.

        If a file path is passed to 'idea', an Idea instance is created from
        that file.

        If a completed Idea instance was passed, it is left intact.

        Raises:
            TypeError: if 'idea' is neither a str nor Idea instance.

        """
        if isinstance(self.idea, str):
            self.idea = Idea(options = self.idea)
        elif isinstance(self.idea, Idea):
            pass
        else:
            error = 'idea must be str or Idea instance'
            raise TypeError(error)
        # Injects base class with 'idea' instance.
        SimpleClass.idea = self.idea
        return self

    def _draft_ingredients(self) -> None:
        """Completes an Ingredients instance for the 'ingredients' attribute.

        If 'ingredients' is a data container, it is assigned to 'df' in a new
            instance of Ingredients.
        If 'ingredients' is a file path, the file is loaded into a DataFrame
            and assigned to 'df' in a new Ingredients instance.
        If 'ingredients' is None, a new Ingredients instance is created and
            assigned to 'ingreidents' with no attached DataFrames.

        Raises:
            TypeError: if 'ingredients' is neither a str, None, DataFrame,
                Series, numpy array, or Ingredients instance.

        """
        if self.ingredients is None:
            self.ingredients = Ingredients()
        elif (isinstance(self.ingredients, pd.Series)
                or isinstance(self.ingredients, pd.DataFrame)
                or isinstance(self.ingredients, np.ndarray)):
            self.ingredients = Ingredients(df = self.ingredients)
        elif isinstance(self.ingredients, str):
            if os.path.isfile(self.ingredients):
                self.ingredients = Ingredients(df = self.depot.load(
                    folder = self.depot.data,
                    file_name = self.ingredients))
            elif os.path.isdir(self.ingredients):
                self.depot.create_glob(folder = self.ingredients)
                self.ingredients = Ingredients()
        else:
            error = ' '.join('ingredients must be a str, DataFrame, None,',
                             'numpy array, or, an Ingredients instance')
            raise TypeError(error)
        return self


    def _get_parameters(self, step: str) -> None:
        """Returns appropriate parameters for subpackage publish method called.

        Args:
            step (str): name of subpackage for which parameters are sought.

        Returns
            parameters (dict): parameters for step.

        """
        parameters = {}
        for parameter in self.publish_parameters[step]:
            parameters[parameter] = getattr(self, parameter)
        return parameters

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        # Sets core attributes and completes appropriate class instances.
        self.core_attributes = ['idea', 'depot', 'ingredients']
        for attribute in self.core_attributes:
            getattr(self, ''.join(['_draft_', attribute]))()
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
            'actuary': ('ingredients'),
            'critic': ('ingredients', 'chef.recipes'),
            'artist': ('ingredients', 'chef.recipes', 'critic.reviews')}
        return self

    def publish(self, ingredients: [Ingredients, pd.DataFrame, pd.Series,
                                    np.ndarray, str, None] = None) -> None:
        """Applies steps in 'order' to 'ingredients'.

        Args:
            ingredients (Ingredients, DataFrame, Series, ndarray, or str): an
                instance of Ingredients, a string containing the full file path
                of where a data file for a pandas DataFrame or Series is
                located, a string containing a,file name in the default data
                folder, as defined in the shared Depot instance, a DataFrame, a
                Series, or numpy ndarray. If a DataFrame, ndarray, or string is
                provided, the resultant DataFrame is stored at the 'df'
                attribute in a new Ingredients instance. If Ingredients is not
                passed, then the local 'ingredients' attribute will be used.

        """
        if ingredients:
            self.ingredients = ingredients
            self._check_ingredients()
        for step in self.order:
            technique =
            self.add_techniques(techniques = self._import_option(
                settings = self.options[step])()
            parameters = self._get_parameters(step = step)
            self.ingredients = getattr(self, step).publish(**parameters)
        return self
