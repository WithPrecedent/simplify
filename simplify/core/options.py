"""
.. module:: options
:synopsis: siMpLify options container
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from configparser import ConfigParser
from dataclasses import dataclass
from importlib import import_module
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd


@dataclass
class SimpleOptions(object):
    """Stores options for siMpLify classes.

    The 'options' attribute is the core of dynamism and extensibility for most
    siMpLify classes. This class stores those options in a dictionary
    also called 'options'. However, because of the various dunder methods,
    the 'options' attribute should not ordinarily be accessed. Instead,
    dictionary and other typical methods applied to the class instance
    will be applied to the stored 'options' dict instead.

    Args:
        options (Dict[str, Any]): dictionary containing options to be stored
             in the class. This object will be stored in the 'options'
             attribute. Options can either be a 1-level dict or 2-level nested
             dict. Further depth of 'options' will not be explored by the
             methods in the class instance.
        parent (SimpleComposite): SimpleComposite instance storing the options
            instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        infer_types (Optional[bool]): whether values in 'options' are converted
            to other datatypes (True) or left as is (False).

    """
    options: Dict[str, Any]
    parent: 'SimpleComposite'
    name: Optional[str] = 'options'
    infer_types: bool = False

    def __post_init__(self) -> None:
        self.update(options = self.options)
        if self.infer_types:
            self.update(options = self._infer_types(options = self.options))
        return self

    """ Dunder Methods """

    def __add__(self,
            other: Union[Dict[str, Any], str, 'SimpleComposite']) -> None:
        """Adds 'options' to the class instance 'options' attribute.

        Args:
            options (Union[Dict, str, SimpleComposite]): can either be a dict,
                a str file path to an ini or py file with options, or a
                SimpleComposite instance with an options attribute.

        """
        self.update(options = other)
        return self

    def __contains__(self, item: str) -> bool:
        """Returns whether item is in 'options'.

        Args:
            item (str): key to be checked for a match in 'options'.

        """
        return item in self.options

    def __delitem__(self, key: str) -> None:
        """Removes a dict section or key if 'key' matches section or key.

        Args:
            key (str): the name of the dictionary key or section to be deleted.

        Raises:
            KeyError: if 'key' not in 'options'.

        """
        try:
            del self.options[key]
        except KeyError:
            found_match = False
            for section in list(self.options.keys()):
                try:
                    del section[key]
                    found_match = True
                except KeyError:
                    pass
        if not found_match:
            error = ' '.join([key, 'not found in', self.parent.name, 'options'])
            raise KeyError(error)
        return self

    def __getattr__(self, attribute: str) -> Any:
        """Intercepts dict methods and applies them to the 'options' attribute.

        Also, 'default' and 'all' are defacto properties which either return
        an appropriate list of options.

        Args:
            attribute (str): attribute sought.

        Returns:
            dict method applied to 'options' or attribute, if attribute exists.

        Raises:
            AttributeError: if attribute  does not exist.

        """
        # Intecepts dict methods and applies them to 'options'.
        if attribute in [
                'clear', 'items', 'pop', 'keys', 'values', 'get', 'fromkeys',
                'setdefault', 'popitem', 'copy']:
            return getattr(self.options, attribute)
        elif attribute in ['all']:
            return list(self.options.keys())
        elif attribute in ['default']:
            if '_default' in self.__dict__:
                return self.__dict__['_default']
            else:
                return list(self.options.keys())
        else:
            error = ' '.join([attribute, 'not found in', self.parent.name,
                              'options'])
            raise AttributeError(error)

    def __getitem__(self, key: str) -> Union[Dict[str, Any], Any]:
        """Returns a section of 'options' or key within a section.

        Args:
            key (str): the name of the dictionary key for which the value is
                sought.

        Returns:
            Union[Dict[str, Any], Any]: dict if 'key' matches a section in
                'options'. If 'key' matches a key within a section, the value,
                which can be any of the supported datatypes is returned.

        Raises:
            KeyError: if 'key' not in 'options'.

        """
        try:
            return self.options[key]
        except KeyError:
            for section in list(self.options.keys()):
                try:
                    return section[key]
                    break
                except KeyError:
                    pass
            error = ' '.join(
                [key, 'not found in', self.parent.name, 'options'])
            raise KeyError(error)


    def __iadd__(self,
                other: Union[Dict[str, Any], str, 'SimpleComposite']) -> None:
        """Adds 'options' to the class instance 'options' attribute.

        Args:
            options (Union[Dict, str, SimpleComposite]): can either be a dict,
                a str file path to an ini or py file with options, or a
                SimpleComposite instance with an options attribute.

        Raises:
            TypeError: if 'options' is neither a dict, str, nor SimpleComposite
                instance.

        """
        self.update(options = other)
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable 'options' dict items()."""
        return iter(self.options.items())

    def __len__(self):
        """Returns length of 'options'."""
        try:
            return len(self.options)
        except AttributeError:
            self.options = {}
            return 0

    def __radd__(self,
                other: Union[Dict[str, Any], str, 'SimpleComposite']) -> None:
        """Adds 'options' to the class instance 'options' attribute.

        Args:
            options (Union[Dict, str, SimpleComposite]): can either be a dict,
                a str file path to an ini or py file with options, or a
                SimpleComposite instance with an options attribute.

        Raises:
            TypeError: if 'options' is neither a dict, str, nor SimpleComposite
                instance.

        """
        self.update(new_settings = options)
        return self

    # def __repr__(self) -> Dict[str, Any]:
    #     """Returns 'options' dict."""
    #     return self.__str__()

    def __setitem__(self, section: str, dictionary: Dict[str, Any]) -> None:
        """Creates new key/value pair(s) in a specified section of
        'options'.

        Args:
            section (str): name of a section in 'options'.
            dictionary (Dict): the dictionary to be placed in that section.

        Raises:
            TypeError if 'section' isn't a str or 'dictionary' isn't a dict.

        """
        try:
            self.options[section].update(dictionary)
        except KeyError:
            try:
                self.options[section] = dictionary
            except TypeError:
                try:
                    self.options[section] = dictionary
                except TypeError:
                    error = ' '.join(['section must be str and dictionary',
                                     'must be dict type'])
                    raise TypeError(error)
        return self

    # def __str__(self) -> Dict[str, Any]:
    #     """Returns 'options' dict."""
    #     return self.options

    """ Private Methods """

    def _infer_types(self,
                options: Union[Dict[str, Any],
                               Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Converts stored values to appropriate datatypes.

        Args:
            options (Dict[str, Any]): can either be a 1 or 2-level dict.

        Returns:
            Dict[str, Any]: with the end
                values converted to supported types.

        """
        new_options = {}
        for section, dictionary in options.items():
            try:
                for key, value in dictionary.items():
                    if section not in new_options:
                        new_options[section] = {key: self._typify(value)}
                    else:
                        new_options[section][key] = self._typify(value)
            except TypeError:
                new_options[section][key] = self._typify(value)
        return new_options

    @staticmethod
    def _numify(variable: str) -> Union[int, float, str]:
        """Attempts to convert 'variable' to a numeric type.

        Args:
            variable (str): variable to be converted.

        Returns
            variable (int, float, str) converted to numeric type, if possible.

        """
        try:
            return int(variable)
        except ValueError:
            try:
                return float(variable)
            except ValueError:
                return variable

    def _load_from_csv(self, file_path: str) -> Dict[str, Any]:
        """Creates a options dict from an .csv file.

        Args:
            file_path (str): path to .csv file.

        Returns:
            Dict[str, Any] of options.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        options = pd.read_csv(file_path, dtype = 'str')
        return options.to_dict(orient = 'list')

    def _load_from_ini(self, file_path: str) -> Dict[str, Any]:
        """Creates a options dict from an .ini file.

        Args:
            file_path (str): path to configparser-compatible .ini file.

        Returns:
            Dict[str, Any] of options.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        try:
            options = ConfigParser(dict_type = dict)
            options.optionxform = lambda option: option
            options.read(file_path)
            options = dict(options._sections)
        except FileNotFoundError:
            error = ' '.join(['options file ', file_path, ' not found'])
            raise FileNotFoundError(error)
        return options

    def _load_from_py(self, file_path: str) -> Dict[str, Any]:
        """Creates a options dictionary from an .py file.

        Args:
            file_path (str): path to python module with 'options' dict defined.

        Returns:
            Dict[str, Any] of options.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        try:
            return getattr(import_module(file_path), 'options')
        except FileNotFoundError:
            error = ' '.join(['options file ', file_path, ' not found'])
            raise FileNotFoundError(error)

    def _typify(self, variable: str) -> Union[List, int, float, bool, str]:
        """Converts stingsr to appropriate, supported datatypes.

        The method converts strings to list (if ', ' is present), int, float,
        or bool datatypes based upon the content of the string. If no
        alternative datatype is found, the variable is returned in its original
        form.

        Args:
            variable (str): string to be converted to appropriate datatype.

        Returns:
            variable (str, list, int, float, or bool): converted variable.
        """
        if (', ') in variable:
            variable = variable.split(', ')
            return [self._numify(v) for v in variable]
        elif re.search('\d', variable):
            return self._numify(variable)
        elif variable in ['True', 'true', 'TRUE']:
            return True
        elif variable in ['False', 'false', 'FALSE']:
            return False
        elif variable in ['None', 'none', 'NONE']:
            return None
        else:
            return variable

    """ Public Methods """

    def set_defaults(default_options: List[str]) -> None:
        """Sets options that will be returned from 'default' attribute."""
        self._defaults = default_options
        return self

    """ Dictionary Compatibility Methods """

    def update(self,
            options: Union[Dict[str, Any], str, 'SimpleComposite']) -> None:
        """Adds 'options' to the class instance 'options' attribute.

        Args:
            options (Union[Dict, str, SimpleComposite]): can either be a dict,
                a str file path to an ini or py file with options, or a
                SimpleComposite instance with an options attribute.

        Raises:
            TypeError: if 'options' is neither a dict, str, nor SimpleComposite
                instance.

        """
        try:
            self.options.update(options)
        except (AttributeError, ValueError, TypeError):
            try:
                self.options.update(options.options)
            except AttributeError:
                extension = str(Path(self.options).suffix)[1:]
                try:
                    self.options = getattr(self,
                            '_'.join(['_load_from', extension]))(
                                file_path = self.options)
                except TypeError:
                    error = 'options must be dict, str, or SimpleComposite'
                    raise TypeError(error)
                except AttributeError:
                    error = ' '.join(
                        [extension, 'is not a supported file type for options'])
                    raise TypeError(error)
        return self