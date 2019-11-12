"""
.. module:: base
:synopsis: abstract base class for siMpLify project
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simplify.core.utilities import listify


@dataclass
class SimpleComposite(ABC):
    """Base class to support a common architecture and sharing of methods.

    SimpleComposite implements a composite tree pattern for organizing
    SimplePlanners and SimpleTechniques.

    SimpleComposite creates a code structure patterned after the writing
    process. It divides processes into four stages which are the names or
    prefixes to the core methods used throughout the siMpLify project:

        1) research: injects needed information from other classes.
        2) draft: sets default attributes (required).
        3) edit: makes any desired changes to the default attributes (optional).
        4) publish: applies selected options to passed argumnts (required).

    A subclass's 'research' and 'draft' methods are automatically called when a
    class is instanced. Any 'edit' methods are optional, but should be called
    next. The 'publish' methods must be called with appropriate arguments
    passed.

    Args:
        These arguments and attributes are not required for any SimpleComposite,
        but are commonly used throughout the project. Brief descriptions are
        included here:

        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        idea (Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located. Once an
            Idea instance is createds, it is automatically an attribute to all
            other SimpleComposite subclasses that are instanced in the future.
        depot (Depot or str): an instance of Depot or a string containing the
            full path of where the root folder should be located for file
            output. A Depot instance contains all file path and import/export
            methods for use throughout the siMpLify project. Once a Depot
            instance is created, it is automatically an attribute of all other
            SimpleComposite subclasses that are instanced in the future.
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
        return self

    """ Private Methods """

    def _convert_wildcards(self, value: Union[str, List[str]]) -> List[str]:
        """Converts 'all', 'default', or 'none' values to a list of items.

        Args:
            value (str or list(str)): name(s) of techniques or projects.

        Returns:
            If 'all', either the 'all' property or all keys listed in 'options'
                dictionary are returned.
            If 'default', either the 'defaults' property or all keys listed in
                'options' dictionary are returned.
            If some variation of 'none', 'none' is returned.
            Otherwise, 'value' is returned intact.

        """
        if value in ['all', ['all']]:
            return self.all
        elif value in ['default', ['default']]:
            self.default
        elif value in ['none', ['none'], 'None', ['None'], None]:
            return ['none']
        else:
            return value

    def _exists(self, attribute: str) -> bool:
        """Returns if attribute exists in subclass and is not None.

        Args:
            attribute (str): name of attribute to be evaluated.

        Returns:
            bool: indicating whether the attribute exists and is not None.

        """
        return (hasattr(self, attribute)
                and getattr(self, attribute) is not None)

    """ Import/Export Methods """

    def load(self,
            name: Optional[str] = None,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> None:
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

    def save(self,
            variable: Optional['SimpleComposite'] = None,
            file_path: Optional[str] = None,
            folder: Optional[str]  = None,
            file_name: Optional[str]  = None,
            file_format: Optional[str]  = None) -> None:
        """Exports a variable or attribute to disc.

        For any arguments not passed, default values stored in the shared Depot
        instance will be used based upon the current 'stage' of the siMpLify
        project.

        Args:
            variable (Optional[SimpleComposite]): a python object or a string
                corresponding to a subclass attribute which should be saved to
                disc.
            file_path (str): a complete file path for the file to be saved.
            folder (str): a path to the folder where the file should be saved
                (not used if file_path is passed).
            file_name (str): contains the name of the file to be saved without
                the file extension (not used if file_path is passed).
            file_format (str): name of file format in Depot.extensions.

        """
        # If a string, 'variable' is converted to a local attribute with the
        # string as its name.
        if variable is None:
            variable = self
        else:
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

    def add_planner(self, planner: 'SimpleComposite') -> None:
        """Sets 'planner' attribute to 'planner'."""
        self.planner = planner
        return self

    def add_techniques(self,
            techniques: Union[List['SimpleComposite'], Dict[str, 'SimpleComposite'],
                              'SimpleComposite'],
            names: Optional[Union[List[str], str]] = None) -> None:
        """Adds technique class instances to class instance.

        Args:
            techniques (Union[List[SimpleComposite], Dict[str, SimpleComposite],
                SimpleComposite): list, dict, or a single SimpleComposite to be added
                to the 'techniques' attribute. If a 'list' or SimpleComposite is
                passed, a conspondng list or name should be passed to 'names'.
            names (Optional[Union[List[str], str]]): name(s) of passed
                techniques to be used as keys in updating the 'techniques' dict.
                Defaults to None.

        """
        if names is not None:
            techniques = dict(zip(names, techniques))
        for key, technique in techniques.items():
            technique.planner = self
            self._inject_attributes(
                attributes = list(self.shared.keys()),
                source = self,
                target = technique)
            self.techniques[key] = technique
        return self

    def remove_planner(self) -> None:
        """Sets 'planner' to None."""
        self.planner = None
        return self

    def remove_techniques(self, techniques: Union[List[str], str]) -> None:
        """Delinks 'techniques' from class instance.

        Args:
            techniques (List[str], str): technique class names to be
                delinked from the current class instance.

        """
        for name in listify(techniques):
            try:
                self.techniques[name].planner = None
                del self.techniques[name]
            except KeyError:
                pass
        return self

    """ Core siMpLify Methods """

    def research(self,
            distributors: Union[List['SimpleComposite'],
                                'SimpleComposite'] = None) -> None:
        for distributor in listify(distributors):
            self = distributor.publish(instance = self)
        return self

    @abstractmethod
    def draft(self) -> None:
        """Required method that sets default values.

        A dict called 'options' may be defined here for subclasses to use
        much of the functionality of SimpleComposite.

        """
        return self

    def edit(self,
            keys: Optional[Union[str, List[str]]] = None,
            values: Optional[Union[Any, List[Any]]] = None,
            options: Optional[Dict[str, Any]] = None) -> None:
        """Updates 'options' dictionary with passed arguments.

        Args:
            keys (str or List[str]): name or list of names for keys to be added
                to 'options'. 'keys' should be the same length as 'values'.
                Default is None.
            values (Any or List[Any]): options which can be integrated in the
                project framework. 'values' should be same length as 'keys'.
                Default is None.
            options (Dict[str, Any]): a dictionary with string keys to
                different siMpLify compatible options. Default is None.

        """
        try:
            self.options.update(options)
        except AttributeError:
            self.options = options
        except TypeError:
            pass
        if len(keys) == len(values):
            try:
                self.options.update(dict(zip(keys, values)))
            except TypeError:
                pass
        return self

    @abstractmethod
    def publish(self) -> None:
        """Required method which applies methods to passed data.

        Subclasses should provide their own 'publish' methods.

        """
        return self