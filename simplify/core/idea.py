"""
.. module:: idea
:synopsis: converts an idea into a siMpLify project
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from configparser import ConfigParser
from dataclasses import dataclass
from importlib import import_module
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.distributor import SimpleDistributor
from simplify.core.options import SimpleOptions
from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify


@dataclass
class Idea(SimpleDistributor):
    """Converts a data science idea into python.

    If 'options' are imported from a file, Idea creates a nested dictionary,
    converting dictionary values to appropriate datatypes, and stores portions
    of the 'options' dictionary as attributes in other classes. Idea is based
    on python's ConfigParser. It seeks to cure some of the shortcomings of the
    base ConfigParser including:
        1) All values in ConfigParser are strings by default.
        2) The nested structure for getting items creates verbose code.
        3) It uses OrderedDict (python 3.6+ orders regular dictionaries).

    To use the Idea class, the user can either pass to 'options':
        1) a file path, which will automatically be loaded into Idea;
        2) a file name which is located in the current working directory,
            which will automatically be loaded into Idea;
                                or,
        3) a prebuilt ConfigParser compatible nested dictionary.

    If 'infer_types' is set to True (the default option), the dictionary values
    are automatically converted to appropriate datatypes (str, list, float,
    bool, and int are currently supported)

    Whichever option is chosen, the nested Idea dictionary is stored in the
    attribute 'options'. However, dictionary access methods can either be
    applied to the 'options' dictionary (e.g., idea.options['general']) or an
    Idea instance (e.g., idea['general']). If using the dictionary 'update'
    method, it is better to apply it to the Idea instance because the Idea
    method is more flexible in handling different kinds of arguments.

    Users can add any key/value pairs from a section of the 'options'
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
        chef_techniques = split, reduce, model

    'verbose' and 'file_type' will automatically be added to every siMpLify
    class because they are located in the 'general' section. If a subclass
    wants attributes from the files section, then the following line should
    appear in __post_init__ before calling super().__post_init__:

        self.idea_sections = ['files']

    If the subclass wants the chef settings as well, then the code should be:

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
                self.chef_techniques = ['split', 'reduce', 'model']
                return self

    Regardless of the idea_sections added, all Idea settings can be similarly
    accessed using dict keys or local attributes. For example:

        self.idea['general']['seed'] # typical dict access technique

        self.idea['seed'] # if no section or other key is named 'seed'

        self.seed # works because 'seed' is in the 'general' section

                            all return 43.

    Within the siMpLify ecosystem, settings of two types take on particular
    importance:
        'parameters': sections with the suffix '_parameters' are automatically
            linked to classes where the prefix matches the class's 'name' or
            'technique'.
        'techniques': settings with the suffix '_techniques' are used to create
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
            in 'options'. If a file name is provided, Idea will look for
            it in the current working directory and store its contents in
            'options'. If a dict is provided, it should be nested into
            sections with individual settings in key/value pairs.
        infer_types (bool): whether values in 'options' are converted to
            other datatypes (True) or left as strings (False).

    """
    configuration: Dict[str, Any]
    name: Optional[str] = 'idea'
    infer_types: Optional[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.draft()
        return self

    """ Dunder Methods """

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

    # def __getattr__(self, attribute: str) -> Any:
    #     """Intercepts dict methods and applies them to the 'options' attribute.

    #     Also, 'default' and 'all' are defacto properties which either return
    #     an appropriate list of options.

    #     Args:
    #         attribute (str): attribute sought.

    #     Returns:
    #         dict method applied to 'options' or attribute, if attribute exists.

    #     Raises:
    #         AttributeError: if attribute  does not exist.

    #     """
    #     # Intecepts dict methods and applies them to 'options'.
    #     if attribute in [
    #             'clear', 'items', 'pop', 'keys', 'values', 'get', 'fromkeys',
    #             'setdefault', 'popitem', 'copy']:
    #         return getattr(self.options, attribute)
    #     else:
    #         error = ' '.join([attribute, 'not found in', self.parent.name,
    #                           'options'])
    #         raise AttributeError(error)

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

    def _are_parameters(self, instance: 'SimpleComposite', section: str) -> bool:
        """Returns whether value stores matching parameters for instance.

        Args:
            instance (SimpleComposite): a class instance to which attributes should
                be added.
            section (str): name of a section of the configuration settings.

        Returns:
            bool: whether the section includes parameters and if those
                parameters correspond to the class name or technique name.

        """
        if '_parameters' in section:
            try:
                return (instance.name == '_'.join(
                    [instance.name, '_parameters']))
            except AttributeError:
                try:
                    return (instance.technique == '_'.join(
                        [instance.technique, '_parameters']))
                except AttributeError:
                    pass
            return False
        else:
            return False

    def _set_sections(self,
            instance: 'SimpleComposite',
            sections: Optional[Union[List[str], str]]) -> List[str]:
        """Finalizes list of sections to be injected into class.

        Args:
            instance (SimpleComposite): a class instance to which attributes should
                be added.
            sections (Optional[Union[List[str], str]]): the sections of the
                configuration that should be stored as local attributes in the
                passed instance.

        """
        if sections is None:
            sections = []
        sections.append('general')
        try:
            sections.extend(listify(instance.idea_sections))
        except AttributeError:
            pass
        try:
            sections.append(instance.planner.name)
        except AttributeError:
            pass
        sections.append(instance.name)
        return deduplicate(sections)

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates 'options' dictionary for Idea instance."""
        self.options = SimpleOptions(
            options = self.configuration,
            parent = self,
            infer_types = self.infer_types)
        return self

    def publish(self,
            instance: 'SimpleComposite',
            sections: Optional[Union[List[str], str]] = None,
            override: Optional[bool] = False) -> 'SimpleComposite':
        """Injects attributes from configuration settings into passed instance.

        Args:
            instance (SimpleComposite): a class instance to which attributes should
                be added.
            sections (Optional[Union[List[str], str]]): the sections of the
                configuration that should be stored as local attributes in the
                passed instance. Defaults to None.
            override (Optional[bool]): if True, even existing attributes in
                instance will be replaced by 'options' key/value pairs. If
                False, current values in those similarly-named attributes will
                be maintained (unless they are None). Defaults to False.

        Returns:
            SimpleComposite: instance with attribute(s) added.

        """
        # Every class that accepts an Idea, gets a local attribute of it.
        instance.idea = self
        self._set_sections(instance = instance, sections = sections)
        if sections:
            for section in listify(sections):
                if self._are_parameters(instance = instance, section = section):
                    instance.idea_parameters = self.options[section]
                else:
                    print('test', self.options.options)
                    try:
                        for key, value in self.options[section].items():
                            if not instance._exists(key) or override:
                                setattr(instance, key, value)
                    except KeyError:
                        pass
        return instance
