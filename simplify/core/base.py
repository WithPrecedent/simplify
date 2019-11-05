"""
.. module:: base
:synopsis: abstract base class for siMpLify package
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
import os
from typing import Any, Iterable, List, Union

from more_itertools import unique_everseen
import numpy as np
import pandas as pd


@dataclass
class SimpleClass(ABC):
    """Base class to support a common architecture and sharing of methods.

    Fundementally, SimpleClass is a cross between composite and builder design
    patterns with some added functionality.

    SimpleClass creates a code structure patterned after the writing process.
    It divides processes into three stages which are the names or prefixes to
    the core methods used throughout the siMpLify package:

        1) draft: sets default attributes (required).
        2) edit: makes any desired changes to the default attributes (optional).
        3) publish: applies selected options to passed argumnts (required).

    A subclass's 'draft' method is automatically called when a class is
    instanced. Any 'edit' methods are optional, but should be called next. The
    'publish' methods must be called with appropriate arguments passed.

    Args:
        These arguments and attributes are not required for any SimpleClass, but
        are commonly used throughout the package. Brief descriptions are
        included here:

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
        ingredients (Ingredients, DataFrame, ndarray, or str): an instance of
            Ingredients, a string containing the full file path of where a data
            file for a pandas DataFrame is located, a string containing a
            file name in the default data folder, as defined in the shared Depot
            instance, a DataFrame, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Ingredients instance.

    """
    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not self._exists('name'):
            self.name = self.__class__.__name__.lower()
        # Initializes 'checks' attribute for validation checks to be performed.
        self.checks = []
        # Sets initial values for subclass.
        self.draft()
        # Injects appropriate settings from shared Idea instance.
        if self.name != 'idea':
            self._inject_idea()
        return self

    """ Dunder Methods """

    def __contains__(self, item: str) -> bool:
        """Checks if item is in 'options' attribute.

        Args:
            item (str): item to be searched for in 'options' keys.

        Returns:
            bool: True, if 'item' in 'options' - otherwise False.

        """
        return item in self.options

    def __delitem__(self, item: str) -> None:
        """Deletes item if in 'options' or, if it is an instance attribute, it
        is assigned a value of None.

        Args:
            item (str): item to be deleted from 'options' or attribute to be
                assigned None.

        Raises:
            KeyError: if item is neither in 'options' or an attribute.

        """
        try:
            del self.options[item]
        except KeyError:
            try:
                delattr(self, item)
            except AttributeError:
                error = item + ' is not in ' + self.__class__.__name__
                raise KeyError(error)
        return self

    def __getattr__(self, attribute: str) -> Any:
        """Returns dict methods applied to options attribute if those methods
        are sought from the class instance.

        Args:
            attribute (str): attribute sought.

        Returns:
            dict method applied to 'options' or attribute, if attribute exists.

        Raises:
            AttributeError: if a dunder attribute is sought or attribute does
                not exist.

        """
        # Intecepts dict methods and applies them to 'options'.
        if attribute in ['clear', 'items', 'pop', 'keys', 'values', 'update',
                         'get', 'fromkeys', 'setdefault', 'popitem', 'copy']:
            return getattr(self.options, attribute)

    def __getitem__(self, item: str) -> Any:
        """Returns item if 'item' is in 'options' or is an atttribute.

        Args:
            item (str): item matching 'options' dictionary key or attribute
                name.

        Returns:
            Value for item in 'options', 'item' attribute value, or None if
                neither of those exist.

        """
        try:
            return self.options[item]
        except KeyError:
            try:
                return getattr(self, item)
            except AttributeError:
                return None

    def __iter__(self) -> Iterable:
        """Returns iterable options dict items()."""
        return self.options.items()

    def __len__(self):
        """Returns length of 'options'."""
        try:
            return len(self.options)
        except AttributeError:
            self.options = {}
            return 0

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets item in 'options' dictionary as 'value'.

        Args:
            item (str): name of key to be set in 'options'.
            value (Any): value for 'item' in 'options'.

        """
        self.options[item] = value
        return self

    """ Private Methods """

    def _convert_wildcards(self, value: Union[str, List[str]]) -> List[str]:
        """Converts 'all', 'default', or 'none' values to a list of items.

        Args:
            value (str or list(str)): name(s) of techniques or packages.

        Returns:
            If 'all', either the 'all' property or all keys listed in 'options'
                dictionary are returned.
            If 'default', either the 'defaults' property or all keys listed in
                'options' dictionary are returned.
            If some variation of 'none', 'none' is returned.
            Otherwise, 'value' is returned intact.

        """
        if value in ['all', ['all']]:
            try:
                return self.all
            except AttributeError:
                return list(self.options.keys())
        elif value in ['default', ['default']]:
            try:
                return self.defaults
            except AttributeError:
                return list(self.options.keys())
        elif value in ['none', ['none'], 'None', ['None'], None]:
            return ['none']
        else:
            return value

    def _draft_techniques(self) -> None:
        """Gets 'steps' from 'idea' if 'steps' not passed."""
        if self.techniques is None:
            self.techniques = getattr(self, '_'.join([self.name, 'techniques']))
        self.steps = self._convert_wildcards(value = self.techniques)
        return self

    def _exists(self, attribute: str) -> bool:
        """Returns if attribute exists in subclass and is not None.

        Args:
            attribute (str): name of attribute to be evaluated.

        Returns:
            bool: indicating whether the attribute exists and is not None.

        """
        return (hasattr(self, attribute)
                and getattr(self, attribute) is not None)

    def _inject_base(self, attribute: str,
                     instance: 'SimpleClass' = None) -> None:
        """Injects base class, with attribute so that it is available to other
        modules instanced in the future.

        Args:
            attribute (str): name of attribute for 'instance' to be stored.
            instance (SimpleClass): instance to be stored in base class.

        """
        instance = instance or self
        setattr(SimpleClass, attribute, instance)
        return self

    def _inject_idea(self) -> None:
        """Injects portions of shared Idea instance 'options' to subclass.

        Every siMpLify class gets the 'general' section of the Idea settings.
        Other sections are added according to the 'name' attribute of the
        subclass and the local 'idea_sections' attribute. How the settings are
        injected is dependent on the 'inject' method in an Idea instance.

        """
        sections = ['general']
        try:
            sections.extend(listify(self.idea_sections))
        except AttributeError:
            pass
        sections.append(self.name)
        self = self.idea.inject(instance = self, sections = sections)
        return self

    """ Import/Export Methods """

    def load(self, name = None, file_path = None, folder = None,
             file_name = None, file_format = None) -> None:
        """Loads object from file into subclass attribute ('name').

        For any arguments not passed, default values stored in the shared Depot
        instance will be used based upon the current 'stage' of the siMpLify
        project.

        Args:
            name (str): name of attribute for the file contents to be stored.
            file_path (str): a complete file path for the file to be loaded.
            folder (str): a path to the folder where the file should be loaded
                from (not used if file_path is passed).
            file_name (str): contains the name of the file to be loaded without
                the file extension (not used if file_path is passed).
            file_format (str): name of file format in Depot.extensions.

        """
        setattr(self, name, self.depot.load(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format))
        return self

    def save(self, variable = None, file_path = None, folder = None,
             file_name = None, file_format = None) -> None:
        """Exports a variable or attribute to disc.

        For any arguments not passed, default values stored in the shared Depot
        instance will be used based upon the current 'stage' of the siMpLify
        project.

        Args:
            variable (any): a python object or a string corresponding to a
                subclass attribute which should be saved to disc.
            file_path (str): a complete file path for the file to be saved.
            folder (str): a path to the folder where the file should be saved
                (not used if file_path is passed).
            file_name (str): contains the name of the file to be saved without
                the file extension (not used if file_path is passed).
            file_format (str): name of file format in Depot.extensions.

        """
        # If a string, 'variable' is converted to a local attribute with the
        # string as its name.
        try:
            variable = getattr(self, variable)
        except TypeError:
            pass
        self.depot.save(
            variable = variable,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format)
        return self

    """ Composite Management Methods """

    def add_children(self,
            children: Union[List['SimpleClass'], 'SimpleClass']) -> None:
        """Adds child class instances to class instance.

        Args:
            children (SimpleClass or list(SimpleClass)): child class instances
                to be linked to the current class instance.

        """
        for child in listify(children, use_null = True):
            self._children.append(child)
            child.parent = self
        return self

    def inject_children(self, attribute: str,
                        instance: 'SimpleClass' = None) -> None:
        """Adds 'instance' to child classes at named 'attribute'.

        Args:
            attribute (str): name of attribute for 'instance' to be stored.
            instance (SimpleClass): instance to be stored in child classes.

        """
        for child in self._children:
            setattr(child, attribute, instance)
        return self

    def inject_parent(self, attribute: str,
                      instance: 'SimpleClass' = None) -> None:
        """Adds 'instance' to parent class at named 'attribute'.

        Args:
            attribute (str): name of attribute for 'instance' to be stored.
            instance (SimpleClass): instance to be stored in parent class.

        """
        setattr(parent, attribute, instance)
        return self

    def remove_children(self,
            children: Union[List['SimpleClass'], 'SimpleClass']) -> None:
        """Removes child class instances from class instance.

        Args:
            children (SimpleClass or list(SimpleClass)): child class instances
                to be delinked from the current class instance.

        """
        for child in listify(children):
            self._children.remove(child)
            child.parent = None
        return self

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self) -> None:
        """Required method that sets default values for a subclass.

        A dict called 'options' may be defined here for subclasses to use
        much of the functionality of SimpleClass.

        """
        pass

    def edit(self, keys: Union[str, List[str]] = None,
             values: Union[str, List[str]] = None,
             options: Dict = None) -> 'SimpleClass':
        """Updates 'options' dictionary with passed arguments.

        Args:
            keys (str or list): a string name or list of names for keys in the
                'options' dict.
            values (object or list(object)): siMpLify compatible objects which
                can be integrated in the package framework. If they are custom
                algorithms, they should be subclassed from SimpleAlgorithm to
                ensure compatibility.
            options (dict): a dictionary with keys of techniques and values of
                algorithms. This should be passed if the user has already
                combined some or all 'techniques' and 'algorithms' into a dict.
        """
        try:
            self.options.update(options)
        except AttributeError:
            self.options = options
        except TypeError:
            pass
        try:
            self.options.update(dict(zip(keys, values)))
        except TypeError:
            pass
        return self

    @abstractmethod
    def publish(self, *args, **kwargs) -> None:
        """Required method which creates and/or applies any objects to be
        applied to data or variables.

        Args:
            *args, **kwargs (any): any arguments needed for application of
                the subclass methods.

        """
        pass

    """ Properties """

    @property
    def all(self) -> List[str]:
        try:
            return list(self.options.keys())
        except AttributeError:
            self.options = {}
            return []

    @property
    def children(self) -> List['SimpleClass']:
        try:
            return self._children
        except AttributeError:
            self._children = []
            return self._children

    @children.setter
    def children(self, children: Union[List['SimpleClass'], 'SimpleClass']) -> (
        None):
        self._children = listify(children, use_null = True)
        return self

    @property
    def defaults(self) -> List[str]:
        try:
            return self._defaults
        except AttributeError:
            return list(self.options.keys())

    @defaults.setter
    def defaults(self, techniques: Union[str, List[str]]) -> None:
        self._defaults = listify(techniques, use_null = True)
        return self

    @property
    def parent(self) -> 'SimpleClass':
        try:
            return self._parent
        except AttributeError:
            self._parent = None
            return self._parent

    @parent.setter
    def parent(self, parent: 'SimpleClass') -> None:
        self._parent = parent
        return self

    @property
    def stage(self) -> str:
        """Returns the shared stage for the overall siMpLify package.

        Returns:
            str: active state.

        """
        try:
            return self._stage_state
        except AttributeError:
            self._stage_state = Stage()
            return self._stage_state

    @stage.setter
    def stage(self, new_stage: str) -> None:
        """Sets the shared stage for the overall siMpLify package

        Args:
            new_stage (str): active state.

        """
        try:
            self._stage_state.change(new_stage)
        except AttributeError:
            self._stage_state = Stage()
            self._stage_state.change(new_stage)
        return self


@dataclass
class Stage(SimpleClass):
    """State machine for siMpLify projects.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'state_machine'

    def __post_init__(self) -> None:
        self._idea_sections = ['simplify']
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __repr__(self) -> str:
        """Returns string name of 'state'."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string name of 'state'."""
        return self.state

    """ Private Methods """

    def _get_states(self) -> List[str]:
        """Determines list of possible stages.

        Returns:
            list: states possible based upon user selections.

        """
        states = []
        for stage in self.simplify_steps:
            if stage == 'farmer':
                for step in self.idea['farmer']['farmer_techniques']:
                    states.append(step)
            else:
                states.append(stage)
        return states

    """ State Machine Methods """

    def change(self, new_state: str) -> None:
        """Changes 'state' to 'new_state'.

        Args:
            new_state (str): name of new state matching a string in 'states'.

        Raises:
            TypeError: if new_state is not in 'states'.

        """
        if new_state in self.states:
            self.state = new_state
        else:
            error = new_state + ' is not a recognized stage'
            raise TypeError(error)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        # Gets list of possible states based upon Idea instance options.
        self._get_states()
        # Sets initial state.
        self.state = self.states[0]
        return self

    def publish(self) -> None:
        """ Returns current state.

        __str__ and __repr__ can also be used to get the current stage.

        """
        return self.state

    """ Properties """

    @property
    def state(self) -> str:
        """Returns current state.

        __str__ and __repr__ can also be used to get the current stage.

        """
        return self.state

    @state.setter
    def state(self, new_state: str) -> None:
        """Changes 'state' to 'new_state'."""
        self.change(new_state = new_state)

    @property
    def states(self) -> List[str]:
        """Returns list of possible stages."""
        return self._states or self._get_states()

    @states.setter
    def states(self, new_states: List[str]) -> None:
        """Sets possible stages to 'new_stages'."""
        self._states = listify(new_states, use_null = True)
