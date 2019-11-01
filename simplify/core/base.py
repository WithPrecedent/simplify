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

from more_itertools import unique_everseen
import numpy as np
import pandas as pd


@dataclass
class SimpleClass(ABC):
    """Base class for siMpLify to support a common architecture and sharing of
    methods.

    SimpleClass creates a code structure patterned after the writing process.
    It divides processes into three stages which are the names or prefixes to
    the core methods used throughout the siMpLify package:

        1) draft: sets default attributes (required).
        2) edit: makes any desired changes to the default attributes.
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
        ingredients (Ingredients, DataFrame, or str): an instance of
            Ingredients, a string containing the full file path of where a data
            file for a pandas DataFrame is located, or a string containing a
            file name in the default data folder, as defined in the shared Depot
            instance. If a DataFrame or string is provided, the resultant
            DataFrame is stored at the 'df' attribute in a new Ingredients
            instance.

    """
    def __post_init__(self):
        """Calls initialization methods and sets defaults."""
        # Initializes 'checks' attribute for validation checks to be performed.
        self.checks = []
        # Sets initial values for subclass.
        self.draft()
        # Sets default 'name' attribute if none exists.
        if not self.exists('name'):
            self.name = self.__class__.__name__.lower()
        # Injects selected attributes from shared Idea instance.
        if self.name != 'idea':
            self._inject_idea()
        return self

    """ Dunder Methods """

    def __contains__(self, item):
        """Checks if item is in 'options'.

        Args:
            item (str): item to be searched for in 'options' keys.

        Returns:
            bool: True, if 'item' in 'options' - otherwise False.

        """
        return item in self.options

    def __delitem__(self, item):
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

    def __getattr__(self, attribute):
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
        # else:
        #     try:
        #         return self.__dict__[attribute]
        #     except KeyError:
        #         error = attribute + ' not found in ' + self.__class__.__name__
        #         raise AttributeError(error)

    def __getitem__(self, item):
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

    """ Private Methods """

    def _convert_wildcards(self, value):
        """Converts 'all', 'default', or 'none' values to a list of items.

        Args:
            value (list or str): name(s) of techniques or packages.

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
            return 'none'
        else:
            return value

    # def _inject_base(self, attribute, instance = None):
    #     """Injects base class, with attribute so that it is available to other
    #     modules in the siMpLify package.

    #     Args:
    #         attribute (str): name of attribute for instance to be stored.
    #         instance (SimpleClass): instance to be stored in base class.

    #     """
    #     instance = instance or self
    #     setattr(SimpleClass, attribute, instance)
    #     return self

    def _inject_idea(self):
        """Injects portions of shared Idea instance 'options' to subclass.

        Every siMpLify class gets the 'general' section of the Idea settings.
        Other sections are added according to the 'name' attribute of the
        subclass and the local 'idea_sections' attribute. How the settings are
        injected is dependent on the 'inject' method in an Idea instance.

        """
        sections = ['general']
        try:
            sections.extend(self.listify(self.idea_sections))
        except AttributeError:
            pass
        sections.append(self.name)
        self = self.idea.inject(instance = self, sections = sections)
        return self

    """ Public Tool Methods """

    @staticmethod
    def add_prefix(iterable, prefix):
        """Adds prefix to each item in a list or keys in a dict.

        An underscore is automatically added after the string prefix.

        Args:
            iterable (list or dict): iterable to be modified.
            prefix (str): prefix to be added.
        Returns:
            list or dict with prefixes added.

        """
        try:
            return {prefix + '_' + k: v for k, v in iterable.items()}
        except TypeError:
            return [prefix + '_' + item for item in iterable]

    @staticmethod
    def add_suffix(iterable, suffix):
        """Adds suffix to each item in a list or keys in a dict.

        An underscore is automatically added after the string suffix.

        Args:
            iterable (list or dict): iterable to be modified.
            suffix (str): suffix to be added.
        Returns:
            list or dict with suffixes added.

        """
        try:
            return {k + '_' + suffix: v for k, v in iterable.items()}
        except TypeError:
            return [item + '_' + suffix for item in iterable]

    @staticmethod
    def deduplicate(iterable):
        """Deduplicates list, pandas DataFrame, or pandas Series.

        Args:
            iterable (list, DataFrame, or Series): iterable to have duplicate
                entries removed.

        Returns:
            iterable (list, DataFrame, or Series, same as passed type):
                iterable with duplicate entries removed.
        """
        try:
            return list(unique_everseen(iterable))
        except TypeError:
            return iterable.drop_duplicates(inplace = True)

    # @staticmethod
    # def dictify(keys, values, ignore_values_list = False):
    #     """Creates dict from list of keys and same value or zips two lists.

    #     Args:
    #         keys (list): keys for new dict.
    #         values (any): valuse for all keys in the new dict or list of values
    #             corresponding to list of keys.
    #         ignore_values_list (bool): if value is a list, but the list should
    #             be the value for all keys, set to True.

    #     Returns:
    #         dict with 'keys' as keys and 'values' as all values or zips two
    #             lists together to form a dict.

    #     """
    #     if isinstance(values, str) and ignore_values_list:
    #         return dict.fromkeys(keys, values)
    #     else:
    #         return dict(zip(keys, values))

    def exists(self, attribute):
        """Returns if attribute exists in subclass and is not None.

        Args:
            attribute (str): name of attribute to be evaluated.

        Returns:
            bool: indicating whether the attribute exists and is not None.

        """
        return (hasattr(self, attribute)
                and getattr(self, attribute) is not None)

    @staticmethod
    def is_nested(dictionary):
        """Returns if passed 'dictionary' is nested at least one-level.

        Args:
            dictionary (dict): dict to be tested.

        Returns:
            bool: indicating whether any value in the 'dictionary' is also a
                dict (meaning that 'dictionary' is nested).

        """
        return any(isinstance(d, dict) for d in dictionary.values())

    @staticmethod
    def listify(variable):
        """Stores passed variable as a list (if not already a list).

        Args:
            variable (str or list): variable to be transformed into a list to
                allow proper iteration.

        Returns:
            variable (list): either the original list, a string converted to a
                list, or a list containing 'none' as its only item.

        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]

    @staticmethod
    def stringify(variable):
        """Converts one item list to a string (if not already a string).

        Args:
            variable (list): variable to be transformed into a string.

        Returns:
            variable (str): either the original str, a string pulled from a
                one-item list, or the original list.

        """
        if variable is None:
            return 'none'
        elif isinstance(variable, str):
            return variable
        else:
            try:
                return variable[0]
            except TypeError:
                return variable

    """ Public Input/Output Methods """

    def load(self, name = None, file_path = None, folder = None,
             file_name = None, file_format = None):
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
             file_name = None, file_format = None):
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
            self.depot.save(
                variable = variable,
                file_path = file_path,
                folder = folder,
                file_name = file_name,
                file_format = file_format)
        return

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self):
        """Required method that sets default values for a subclass.

        A dict called 'options' may be defined here for subclasses to use
        much of the functionality of SimpleClass.

        """
        pass

    def edit(self, keys = None, values = None, options = None):
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
    def publish(self, *args, **kwargs):
        """Required method which creates and/or applies any objects to be
        applied to data or variables.

        Args:
            *args, **kwargs (any): any arguments needed for application of
                the subclass methods.

        """
        pass

    """ Properties """

    @property
    def stage(self):
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
    def stage(self, new_stage):
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

    def __post_init__(self):
        self.draft()
        return self

    """ Dunder Methods """

    def __repr__(self):
        """Returns string name of 'state'."""
        return self.__str__()

    def __str__(self):
        """Returns string name of 'state'."""
        return self.state

    """ Private Methods """

    def _get_states(self):
        """Determines list of possible stages.

        Returns:
            list: states possible based upon user selections.

        """
        states = []
        for stage in self.idea.options['simplify']['simplify_techniques']:
            if stage == 'farmer':
                for step in self.idea.options['farmer']['farmer_techniques']:
                    states.append(step)
            else:
                states.append(stage)
        return states

    """ Public State Machine Methods """

    def change(self, new_state):
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

    def draft(self):
        # Gets list of possible states based upon Idea instance options.
        self._get_states()
        # Sets initial state.
        self.state = self.states[0]
        return self

    def publish(self):
        """ Returns current state.

        __str__ and __repr__ can also be used to get the current stage.

        """
        return self.state

    """ Properties """

    @property
    def state(self):
        """Returns current state.

        __str__ and __repr__ can also be used to get the current stage.

        """
        return self.state

    @state.setter
    def state(self, new_state):
        """Changes 'state' to 'new_state'."""
        self.change(new_state = new_state)

    @property
    def states(self):
        """Returns list of possible stages."""
        return self._states or self._get_states()

    @states.setter
    def states(self, new_states):
        """Sets possible stages to 'new_stages'."""
        self._states = new_states
