"""
.. module:: planner
:synopsis: iterable builders and containers
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""


from dataclasses import dataclass
from importlib import import_module
from itertools import product
import os
from typing import Any, List, Dict, Iterable, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from simplify import timer
from simplify.core.base import SimpleClass
from simplify.core.depot import Depot
from simplify.core.idea import Idea
from simplify.core.ingredients import Ingredients
from simplify.core.technique import SimpleTechnique
from simplify.core.utilities import listify


@dataclass
class SimplePlanner(SimpleClass):
    """Base class for building and controlling iterable techniques.

    This class adds methods useful to create iterators and iterate over passed
    arguments based upon user-selected options. SimplePackage subclasses
    construct iterators and process data with those iterators.

    Args:

        idea (Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located. Once an
            Idea instance is createds, it is automatically an attribute to all
            other SimpleClass subclasses that are instanced in the future.
            Required.
        depot (Depot or str): an instance of Depot or a string containing the
            full path of where the root folder should be located for file
            output. A Depot instance contains all file path and import/export
            methods for use throughout the siMpLify package. Once a Depot
            instance is created, it is automatically an attribute of all other
            SimpleClass subclasses that are instanced in the future. Default is
            None.
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
            
    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    idea: Union[Idea, str]
    depot: Union[Depot, str, None] = None
    ingredients: Union[Ingredients, pd.DataFrame, pd.Series, np.ndarray, str,
                       None] = None
    steps: Union[List[str], str] = None
    
    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        try:
            return self.plans.items()
        except AttributeError:
            return {}

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
    
    def _draft_steps(self) -> None:
        """Gets 'steps' from injected Idea setting or sets to empty dict."""
        if self.steps is None:
            try:
                self.steps = getattr(self, '_'.join([self.name, 'techniques']))
            except AttributeError:
                self.steps = {}
        return self
    
    def _draft_techniques(self) -> None:
        """Creates 'techniques' containing technique builder instances."""
        for step in listify(self.steps):
            try:
                instance = self._import_option(settings = self.options[step])()
                self.add_techniques(techniques = instance)
            except KeyError:
                error = ' '.join([step,
                                  'does not match an option in', self.name])
                raise KeyError(error)
        return self

    def _draft_plans(self) -> None:
        """Creates cartesian product of all plans for 'techniques' of child
        'techniques'."""
        plans = []
        for technique in self.technique:
            try:
                plans.append(list(technique.techniques.keys()))
            except AttributeError:
                plans.append(['none'])
        self.plans = list(map(list, product(*plans)))
        return self

    def _import_option(self, settings: Tuple[str, str]) -> object:
        """Lazily loads object from siMpLify module.

        Args:
            settings (Tuple[str: str]): first item in settings is the module to
                import from and the second is the object in that module to
                import.

        Returns:
            imported object.

        """
        return getattr(import_module(settings[0]), settings[1])

    def _publish_plan_metadata(self, number: int) -> Dict[str, Any]:
        metadata = {'number': number + 1}
        try:
            metadata.update(self.metadata)
        except AttributeError:
            pass
        return metadata
    
    def _publish_techniques(self, data: Union[Ingredients, Tuple]) -> None:
        """Finalizes all prepared 'techniques'."""
        for technique in self._techniques.values():
            technique.publish(data = data)
        return self

    def _publish_plans(self) -> None:
        """Converts 'plans' from list of lists to SimplePlan(s)."""
        new_plans = {}
        for i, plan in enumerate(self.plans):
            plan_steps = dict(zip(self.steps, plan))
            plan_steps = self._publish_sequence(steps = plan_steps)
            metadata = self._publish_plan_metadata(number = i)
            try:
                new_plans[str(i + 1)] = self.plan_container(
                    metadata = metadata,
                    steps = plan_steps)
            except AttributeError:
                self.plan_container = SimplePlan
                new_plans[str(i + 1)] = self.plan_container(
                    metadata = metadata,
                    steps = plan_steps)
        self.plans = new_plans
        return self

    def _publish_sequence(self, 
            steps: Dict[str, str]) -> Dict[str, SimpleClass]:
        published_steps = {}
        for step, technique in steps.items():
            published_steps[steps] = self.techniques[technique]
        return published_steps

    def _extra_processing(self, plan: SimpleClass,
            data: SimpleClass) -> Tuple[SimpleClass, SimpleClass]:
        return plan, data

    """ Public Import/Export Methods """

    def load_plan(self, file_path):
        """Imports a single recipe from disc and adds it to the class iterable.

        Args:
            file_path: a path where the file to be loaded is located.
        """
        self.edit_plans(
            plans = self.depot.load(
                file_path = file_path,
                file_format = 'pickle'))
        return self

    """ Core siMpLify methods """

    def draft(self):
        """Creates initial settings for class based upon Idea settings."""
        super().draft()
        core_attributes = ('idea', 'depot', 'ingredients')
        for attribute in core_attributes:
            getattr(self, '_'.join(['_draft', attribute]))()
        self._inject_idea()
        self._draft_steps()
        self._draft_techniques()
        self._draft_plans()
        return self

    def edit_plans(self, plans):
        """Adds a comparer or list of plans to the attribute named in
        'comparer_iterable'.

        Args:
            plans (dict(str/int: SimplePlan or list(dict(str/int:
                SimplePlan)): plan(s) to be added to the attribute named in
                'comparer_iterable'.

        """
        if isinstance(plans, dict):
            plans = list(plans.values())
        try:
            last_num = list(self.plans.keys())[-1:]
        except TypeError:
            last_num = 0
        try:
            for i, comparer in enumerate(listify(plans)):
                self.plans.update({last_num + i + 1: comparer})
        except AttributeError:
            self.plans.update({last_num + i + 1: plans})
        return self

    def publish(self, data: SimpleClass, **kwargs):
        super().publish()
        self._publish_steps(data = data)
        self._publish_plans()
        for number, plan in self.plans.items():
            if self.verbose:
                print('Testing', plan.name, str(number))
            plan.publish(data = data, **kwargs)
            plan, data = self._extra_processing(plan = plan, data = data)
        return self


@dataclass
class SimplePlan(SimpleClass):
    """Iterator for a siMpLify process.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        number (int): number of plan in a sequence - used for recordkeeping
            purposes.
        steps (list(SimpleClass)): any

    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    name: str = 'generic_plan'
    metadata: Dict[str, Any] = None
    steps: Dict[str, SimpleClass] = None

    def __post_init__(self) -> None:
        if self.steps is None:
            self.steps = []
        return self

    """ Dunder Methods """

    def __add__(self, steps: Dict[str, SimpleClass]) -> None:
        """Adds step(s) at the end of 'steps'.

        Args:
            steps (Dict[str: SimpleClass]): the next step(s) to be added.

        """
        self.add(steps = steps)
        return self

    def __iadd__(self, steps: Dict[str, SimpleClass]) -> None:
        """Adds step(s) at the end of 'steps'.

        Args:
            steps (Dict[str: SimpleClass]): the next step(s) to be added.

        """
        self.add(steps = steps)
        return self

    """ Import/Export Methods """

    def load(self, file_path: str = None, folder: str = None, 
            file_name: str = None) -> None:
        """Loads 'steps' from disc.

        For any arguments not passed, default values stored in the shared Depot
        instance will be used based upon the current 'stage' of the siMpLify
        project.

        Args:
            file_path (str): a complete file path for the file to be loaded.
            folder (str): a path to the folder where the file should be loaded
                from (not used if file_path is passed).
            file_name (str): contains the name of the file to be loaded without
                the file extension (not used if file_path is passed).

        """
        self.steps = self.depot.load(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = 'pickle')
        return self

    def save(self, file_path: str = None, folder: str = None, 
             file_name: str = None) -> None:
        """Exports 'steps' to disc.

        For any arguments not passed, default values stored in the shared Depot
        instance will be used based upon the current 'stage' of the siMpLify
        project.

        Args:
            file_path (str): a complete file path for the file to be saved.
            folder (str): a path to the folder where the file should be saved
                (not used if file_path is passed).
            file_name (str): contains the name of the file to be saved without
                the file extension (not used if file_path is passed).

        """
        self.depot.save(
            variable = self.steps,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = 'pickle')
        return self

    """ Steps Methods """

    def add(self, steps: Dict[str, SimpleClass]) -> None:
        """Adds step(s) at the end of 'steps'.

        Args:
            steps: Dict[str: SimpleClass]): the next step(s) to be added.

        """
        self.steps.update(steps)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        return self

    def publish(self, data: SimpleClass, **kwargs) -> None:
        """Applies 'steps' to passed 'data'.

        Args:
            data ('SimpleClass'): a data container or other SimpleClass
                for steps to be applied to.

        """
        setattr(self, data.name, data)
        for step, technique in self.steps.items():
            try:
                self.stage = step
            except KeyError:
                pass
            setattr(self, data.name, technique.publish(data = data, **kwargs))
        return self
