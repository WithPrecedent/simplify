"""
.. module:: idea
:synopsis: converts an idea into python
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from configparser import ConfigParser
from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify


@dataclass
class Idea(object):
    """Converts a data science idea into python.

    If 'configuration' is imported from a file, Idea creates a dictionary,
    converting dictionary values to appropriate datatypes, and stores portions
    of the 'configuration' dictionary as attributes in other classes. Idea is
    based on python's ConfigParser. It seeks to cure some of the shortcomings of
    the base ConfigParser including:
        1) All values in ConfigParser are strings by default.
        2) The nested structure for getting items creates verbose code.
        3) It uses OrderedDict (python 3.6+ orders regular dictionaries).

    To use the Idea class, the user can either pass to 'configuration':
        1) a file path, which will automatically be loaded into Idea;
        2) a file name which is located in the current working directory,
            which will automatically be loaded into Idea;
                                or,
        3) a prebuilt ConfigParser compatible nested dictionary.

    If 'infer_types' is set to True (the default option), the dictionary values
    are automatically converted to appropriate datatypes (str, list, float,
    bool, and int are currently supported)

    Whichever option is chosen, the nested Idea dictionary is stored in the
    attribute 'configuration'. However, dictionary access methods can either be
    applied to the 'configuration' dictionary (e.g.,
    idea.configuration['general']) or an Idea instance (e.g., idea['general']).
    If using the dictionary 'update' method, it is better to apply it to the
    Idea instance because the Idea method is more flexible in handling different
    kinds of arguments.

    Users can add any key/value pairs from a section of the 'configuration'
    dictionary as attributes to a class instance by using the 'inject' method.

    For example, if the idea source file is as follows:

        [general]
        verbose = True
        seed = 43

        [files]
        source_format = csv
        test_data = True
        test_chunk = 500
        random_test_chunk = True

        [chef]
        chef_steps = split, reduce, model

    'verbose' and 'file_type' will automatically be added to every siMpLify
    class because they are located in the 'general' section. If a subclass
    wants attributes from the files section, then the following line should
    appear in __post_init__ before calling super().__post_init__:

        self.idea_sections = ['files']

    If the subclass wants the 'chef' settings as well, then the code should be:

        self.idea_sections = ['files', 'chef']

    If that latter code is included, an equivalent to this class will be
    created:

        class FakeClass(object):

            def __init__(self):
                self.verbose = True
                self.seed = 43
                self.source_format = 'csv'
                self.test_data = True
                self.test_chunk = 500
                self.random_test_chunk = True
                self.chef_steps = ['split', 'reduce', 'model']
                return self

    Regardless of the idea_sections added, all Idea settings can be similarly
    accessed using dict keys or local attributes. For example:

        self.idea['general']['seed'] # typical dict access step

        self.idea['seed'] # if no section or other key is named 'seed'

        self.seed # works because 'seed' is in the 'general' section

                            all return 43.

    Within the siMpLify ecosystem, settings of two types take on particular
    importance:
        'parameters': sections with the suffix '_parameters' are automatically
            linked to classes where the prefix matches the class's 'name' or
            'step'.
        'steps': settings with the suffix '_steps' are used to create
            iterable lists of actions to be taken (whether in parallel or
            serial).

    Because Idea uses ConfigParser, it only allows 2-level dictionaries. The
    desire for accessibility and simplicity dictated this limitation.

    Args:
        name (str): as with other classes in siMpLify, the name is used for
            coordinating between classes. If Idea is subclassed, it is
            generally a good idea to keep the 'name' attribute as 'idea'.
        configuration (str, dict): either a file path, file name, or two-level
            nested dictionary storing settings. If a file path is provided, a
            nested dict will automatically be created from the file and stored
            in 'configuration'. If a file name is provided, Idea will look for
            it in the current working directory and store its contents in
            'configuration'. If a dict is provided, it should be nested into
            sections with individual settings in key/value pairs.
        infer_types (bool): whether values in 'configuration' are converted to
            other datatypes (True) or left as strings (False).

    """
    configuration: Union[Dict[str, Any], str]
    name: Optional[str] = 'idea'
    infer_types: Optional[bool] = True

    def __post_init__(self) -> None:
        self.draft()
        return self

    """ Dunder Methods """

    def __add__(self, other: Union[Dict[str, Any], str]) -> None:
        """Adds 'other to the class instance '__dict__' attribute.

        Args:
            other (Union[Dict[str, Any], str]): can either be a dict or
                a str file path to a supported file type with settings.

        Raises:
            TypeError: if '__dict__' is neither a Dict nor str.

        """
        self.update(settings = other)
        return self

    def __contains__(self, item: str) -> bool:
        """Returns whether item is in 'configuration'.

        Args:
            item (str): key to be checked for a match in 'configuration'.

        """
        return item in self.configuration

    def __delitem__(self, key: str) -> None:
        """Removes a dict section or key if 'key' matches section or key.

        Args:
            key (str): the name of the dictionary key or section to be deleted.

        Raises:
            KeyError: if 'key' not in 'configuration'.

        """
        try:
            del self.configuration[key]
        except KeyError:
            found_match = False
            for section in list(self.configuration.keys()):
                try:
                    del section[key]
                    found_match = True
                except KeyError:
                    pass
        if not found_match:
            error = ' '.join([key, 'not found in Idea'])
            raise KeyError(error)
        return self

    def __getattr__(self, attribute: str) -> Any:
        """Intercepts dict methods and applies them to 'configuration'.

        Also, 'default' and 'all' are defacto properties which either return
        an appropriate list of configuration.

        Args:
            attribute (str): attribute sought.

        Returns:
            dict method applied to 'configuration' or attribute, if attribute
                exists.

        Raises:
            AttributeError: if attribute  does not exist.

        """
        # Intecepts dict methods and applies them to 'configuration'.
        if attribute in [
                'clear', 'items', 'pop', 'keys', 'values', 'get', 'fromkeys',
                'setdefault', 'popitem', 'copy']:
            return getattr(self.configuration, attribute)
        else:
            error = ' '.join([attribute, 'not found in Idea'])
            raise AttributeError(error)

    def __getitem__(self, key: str) -> Union[Dict[str, Any], Any]:
        """Returns a section of 'configuration' or key within a section.

        Args:
            key (str): the name of the dictionary key for which the value is
                sought.

        Returns:
            Union[Dict[str, Any], Any]: dict if 'key' matches a section in
                'configuration'. If 'key' matches a key within a section, the
                value, which can be any of the supported datatypes is returned.

        Raises:
            KeyError: if 'key' not in 'configuration'.

        """
        try:
            return self.configuration[key]
        except KeyError:
            for section in list(self.configuration.keys()):
                try:
                    return section[key]
                    break
                except KeyError:
                    pass
            error = ' '.join(
                [key, 'not found in Idea'])
            raise KeyError(error)

    def __iadd__(self, other: Union[Dict[str, Any], str]) -> None:
        """Adds 'other to the class instance '__dict__' attribute.

        Args:
            other (Union[Dict[str, Any], str]): can either be a dict or
                a str file path to a supported file type with configuration.

        Raises:
            TypeError: if '__dict__' is neither a Dict nor str.

        """
        self.update(configuration = other)
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable 'configuration' dict."""
        return iter(self.configuration)

    def __len__(self):
        """Returns length of 'configuration' dict."""
        return len(self.configuration)

    def __radd__(self, other: Union[Dict[str, Any], str]) -> None:
        """Adds 'other to the class instance '__dict__' attribute.

        Args:
            other (Union[Dict[str, Any], str]): can either be a dict or
                a str file path to a supported file type with configuration.

        Raises:
            TypeError: if '__dict__' is neither a Dict nor str.

        """
        self.update(configuration = other)
        return self

    def __repr__(self) -> Dict[str, Any]:
        """Returns 'configuration' dict."""
        return self.__str__()

    def __setitem__(self, section: str, dictionary: Dict[str, Any]) -> None:
        """Creates new key/value pair(s) in a specified section of
        'configuration'.

        Args:
            section (str): name of a section in 'configuration'.
            dictionary (Dict): the dictionary to be placed in that section.

        Raises:
            TypeError if 'section' isn't a str or 'dictionary' isn't a dict.

        """
        try:
            self.configuration[section].update(dictionary)
        except KeyError:
            try:
                self.configuration[section] = dictionary
            except TypeError:
                try:
                    self.configuration[section] = dictionary
                except TypeError:
                    error = ' '.join(['section must be str and dictionary',
                                     'must be dict type'])
                    raise TypeError(error)
        return self

    def __str__(self) -> Dict[str, Any]:
        """Returns 'configuration' dict."""
        return self.configuration

    """ Private Methods """

    def _are_parameters(self,
            instance: 'SimpleContributor', section: str) -> bool:
        """Returns whether value stores matching parameters for instance.

        Args:
            instance (SimpleContributor): a class instance to which attributes
                should be added.
            section (str): name of a section of the configuration settings.

        Returns:
            bool: whether the section includes parameters and if those
                parameters correspond to the class name or step name.

        """
        if '_parameters' in section:
            try:
                return (instance.name == '_'.join(
                    [instance.name, '_parameters']))
            except AttributeError:
                try:
                    return (instance.step == '_'.join(
                        [instance.step, '_parameters']))
                except AttributeError:
                    pass
            return False
        else:
            return False

    def _infer_types(self,
            configuration: Dict[str,Dict[str, Any]]) -> (
                Dict[str,Dict[str, Any]]):
        """Converts stored values to appropriate datatypes.

        Args:
            configuration (Dict[str,Dict[str, Any]]): 2-level nested dict.

        Returns:
            Dict[str,Dict[str, Any]]: with the end values converted to supported
                types.

        """
        new_configuration = {}
        for section, dictionary in configuration.items():
            for key, value in dictionary.items():
                try:
                    new_configuration[section][key] = self._typify(value)
                except KeyError:
                    new_configuration[section] = {key: self._typify(value)}
        return new_configuration

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
        """Creates a configuration dict from an .csv file.

        Args:
            file_path (str): path to .csv file.

        Returns:
            Dict[str, Any] of settings.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        configuration = pd.read_csv(file_path, dtype = 'str')
        return configuration.to_dict(orient = 'list')

    def _load_from_ini(self, file_path: str) -> Dict[str, Any]:
        """Creates a configuration dict from an .ini file.

        Args:
            file_path (str): path to configparser-compatible .ini file.

        Returns:
            Dict[str, Any] of configuration.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        try:
            configuration = ConfigParser(dict_type = dict)
            configuration.optionxform = lambda option: option
            configuration.read(file_path)
            configuration = dict(configuration._sections)
        except FileNotFoundError:
            error = ' '.join(['configuration file ', file_path, ' not found'])
            raise FileNotFoundError(error)
        return configuration

    def _load_from_py(self, file_path: str) -> Dict[str, Any]:
        """Creates a configuration dictionary from an .py file.

        Args:
            file_path (str): path to python module with '__dict__' dict defined.

        Returns:
            Dict[str, Any] of configuration.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        try:
            return getattr(import_module(file_path), '__dict__')
        except FileNotFoundError:
            error = ' '.join(['configuration file ', file_path, ' not found'])
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

    def _set_sections(self,
            instance: 'SimpleContributor',
            sections: Optional[Union[List[str], str]]) -> List[str]:
        """Finalizes list of sections to be injected into class.

        Args:
            instance (SimpleContributor): a class instance to which attributes
                should be added.
            sections (Optional[Union[List[str], str]]): the sections of the
                configuration that should be stored as local attributes in the
                passed instance.

        """
        if sections is None:
            sections = []
        else:
            sections = listify(sections)
        sections.append('general')
        try:
            sections.extend(listify(instance.idea_sections))
        except AttributeError:
            pass
        try:
            sections.append(instance.book.name)
        except AttributeError:
            pass
        try:
            sections.append(instance.chapter.book.name)
        except AttributeError:
            pass
        sections.append(instance.name)
        return deduplicate(sections)

    """ Dictionary Compatibility Methods """

    def update(self,
            configuration: Union[Dict[str,Dict[str, Any]],
                                 str, 'Idea']) -> None:
        """Adds 'configuration' to the class instance 'configuration' attribute.

        Args:
            configuration (Union[Dict[str,Dict[str, Any]], str, 'Idea']): can
                either be a dict, a str file path to an ini or py file with
                configuration, or an Idea instance with a configuration
                attribute.

        Raises:
            TypeError: if 'configuration' is neither a dict, str, nor Idea
                instance.

        """
        try:
            self.configuration.update(configuration)
        except (ValueError, TypeError):
            try:
                self.configuration.update(configuration.configuration)
            except AttributeError:
                try:
                    extension = str(Path(configuration).suffix)[1:]
                    self.configuration.update(getattr(self,
                            '_'.join(['_load_from', extension]))(
                                file_path = configuration))
                except TypeError:
                    error = 'configuration must be dict, str, or Idea'
                    raise TypeError(error)
                except AttributeError:
                    error = ' '.join(
                        [extension, 'is not a supported file type for Idea'])
                    raise TypeError(error)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates 'configuration' dictionary from passed 'configuration'."""
        new_configuration = self.configuration
        self.configuration = {}
        self.update(configuration = new_configuration)
        if self.infer_types:
            self.update(configuration = self._infer_types(
                configuration = self.configuration))
        return self

    def publish(self,
            instance: 'SimpleContributor',
            sections: Optional[Union[List[str], str]] = None,
            override: Optional[bool] = False) -> 'SimpleContributor':
        """Injects attributes from configuration settings into passed instance.

        Args:
            instance (SimpleContributor): a class instance to which attributes
                should be added.
            sections (Optional[Union[List[str], str]]): the sections of the
                configuration that should be stored as local attributes in the
                passed instance. Defaults to None.
            override (Optional[bool]): if True, even existing attributes in
                instance will be replaced by 'configuration' key/value pairs. If
                False, current values in those similarly-named attributes will
                be maintained (unless they are None). Defaults to False.

        Returns:
            SimpleContributor: instance with attribute(s) added.

        """
        # Sets and injections section values into instance.
        sections = self._set_sections(instance = instance, sections = sections)
        if sections:
            for section in listify(sections):
                if self._are_parameters(instance = instance, section = section):
                    instance.idea_parameters = self.configuration[section]
                else:
                    try:
                        for key, value in self.configuration[section].items():
                            if not hasattr(instance, key) or override:
                                setattr(instance, key, value)
                    except KeyError:
                        pass
        return instance
