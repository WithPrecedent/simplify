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
import warnings

from more_itertools import unique_everseen
import numpy as np
import pandas as pd
#from tensorflow.test import is_gpu_available


@dataclass
class SimpleClass(ABC):
    """Parent class for siMpLify package to support a common architecture and
    allow for sharing of universal methods.

    SimpleClass creates a code structure patterned after the writing process.
    It divides processes into three stages which are the names or prefixes to
    the core methods used throughout the siMpLify package:

        1) draft: sets default attributes (required).
        2) edit: makes any desired changes to the default attributes.
        3) publish: creates python objects based upon those attributes.

    A subclass's 'draft' method is automatically called when a class is
    instanced. Any 'edit' methods are optional, but should be called next. The
    'publish' methods must be called with appropriate parameters passed.

    Args:
        These arguments are not required for any SimpleClass, but are commonly
        used throughout the package. Brief descriptions are included here:

        idea (Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located. Once an
            Idea instance is created by a subclass of SimpleClass, it is
            automatically made available to all other SimpleClass subclasses
            that are instanced in the future.
        depot (Depot): an instance of Depot. Once a Depot instance is created by
            a subclass of SimpleClass, it is automatically made available to
            all other SimpleClass subclasses that are instanced in the future.
        ingredients (Ingredients or str): an instance of Ingredients of a string
            containing the full file path of where a supported file type that
            can be loaded into a pandas DataFrame is located. If it is a
            string,  the loaded DataFrame will be bound to a new ingredients
            instance as the 'df' attribute.
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package.

    """

    def __post_init__(self):
        """Calls initialization methods and sets defaults."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Creates SimpleChecker instance to execute various validation and
        # initialization methods and checks attribute for checks to be
        # performed.
        self.checker = SimpleChecker()
        self.checks = []
        # Sets default 'name' attribute if none exists.
        self.checker.publish(checks = ['name'])
        # Sets initial values for subclass.
        self.draft()
        # Creates 'idea' attribute if a string is passed to Idea when subclass
        # was instanced.
        if self.__class__.__name__ == 'Idea':
            self.publish()
            self._inject_base(attribute = 'idea')
        else:
            # Injects parameters and attributes from shared Idea instance.
            self._inject_idea()
        return self

    """ Dunder Methods """

    def __call__(self, idea, *args, **kwargs):
        """When called as a function, a subclass will return the implement
        method after running __post_init__.

        Args:
            idea(Idea or str): an instance of Idea or path where an Idea
                configuration file is located. This argument must be passed
                when  a subclass is called as a function.
            *args and **kwargs(any): passed to the 'implement' method.

        Returns:
            return value of 'implement' method.

        Raises:
            NotImplementedError: if called by Idea instance.
        """
        if self.__class__.__name__ == 'Idea':
            error = 'Idea cannot be called as a function'
            raise NotImplementedError(error)
        else:
            self.idea = idea
            self.__post_init__()
            return self.publish(*args, **kwargs)

    def __contains__(self, item):
        """Checks if item is in 'options'.

        Args:
            item(str): item to be searched for in 'options' keys.

        Returns:
            True, if 'item' in 'options' - otherwise False.

        """
        return item in self.options

    def __delitem__(self, item):
        """Deletes item if in 'options' or, if it is an instance attribute, it
        is assigned a value of None.

        Args:
            item(str): item to be deleted from 'options' or attribute to be
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

    def __getattr__(self, attr):
        """Returns dict methods applied to options attribute if those methods
        are sought from the class instance.

        Args:
            attr(str): attribute sought.

        Returns:
            dict method applied to 'options' or attribute, if attribute exists.

        Raises:
            AttributeError: if a dunder attribute is sought or attribute does
                not exist.

        """
        # Intecepts common dict methods and applies them to 'options'.
        if attr in ['clear', 'items', 'pop', 'keys', 'values']:
            return getattr(self.options, attr)
        else:
            try:
                return self.__dict__[attr]
            except KeyError:
                error = attr + ' not found in ' + self.__class__.__name__
                raise AttributeError(error)

    def __getitem__(self, item):
        """Returns item if 'item' is in 'options' or is an atttribute.

        Args:
            item(str): item matching 'options' dictionary key or attribute
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

    # def __repr__(self):
    #     return self.__str__()

    # def __str__(self):
    #     return self.name

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
            new_stage(str): active state.

        """
        try:
            self._stage_state.change(new_stage)
        except AttributeError:
            self._stage_state = Stage()
            self._stage_state.change(new_stage)
        return self

    """ Private Methods """

    def _convert_wildcards(self, value):
        """Converts 'all', 'default', or 'none' values to a list of items.

        Args:
            value (list or str): name(s) of techniques, steps, or managers.

        Returns:
            If 'all', either the 'all' property or all keys listed in 'options'
                dictionary are returned.
            If 'default', either the 'defaults' property or all keys listed in
                'options' dictionary are returned.
            If some variation of 'none', 'none' is returned.
            Otherwise, value is returned intact.
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

    def _inject_base(self, attribute):
        """Injects parent class, SimpleClass, with attribute so that it is
        available to other modules in the siMpLify getattr(self, attribute).
        """
        setattr(SimpleClass, attribute, self)
        return self

    def _inject_idea(self):
        """Injects portions of Idea instance 'options' to subclass.

        Every siMpLify class gets the 'general' section of the Idea settings.
        Other sections are added according to the 'name' attribute of the
        subclass and the local 'idea_sections' attribute. How the settings are
        injected is dependent on the 'inject' method in an Idea instance.

        """
        # Adds attributes to class from appropriate sections of the idea.
        sections = ['general']
        try:
            sections.extend(self.listify(self.idea_sections))
        except AttributeError:
            pass
        if self.name in self.idea.options and not self.name in sections:
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
            iterable(list, DataFrame, or Series): iterable to have duplicate
                entries removed.

        Returns:
            iterable (list, DataFrame, or Series, same as passed type):
                iterable with duplicate entries removed.
        """
        try:
            return list(unique_everseen(iterable))
        except TypeError:
            return iterable.drop_duplicates(inplace = True)

    @staticmethod
    def dictify(keys, values, ignore_values_list = False):
        """Creates dict from list of keys and same value or zips two lists.

        Args:
            keys (list): keys for new dict.
            values (any): valuse for all keys in the new dict or list of values
                corresponding to list of keys.
            ignore_values_list (bool): if value is a list, but the list should
                be the value for all keys, set to True.

        Returns:
            dict with 'keys' as keys and 'values' as all values or zips two
                lists together to form a dict.
        """
        if isinstance(values, str) and ignore_values_list:
            return dict.fromkeys(keys, values)
        else:
            return dict(zip(keys, values))

    def exists(self, attribute):
        """Returns if attribute exists in subclass and is not None.

        Args:
            attribute(str): name of attribute to be evaluated.

        Returns:
            boolean value indicating whether the attribute exists and is not
                None.
        """
        return (hasattr(self, attribute)
                and getattr(self, attribute) is not None)

    @staticmethod
    def is_nested(dictionary):
        """Returns if passed 'dictionary' is nested at least one-level.

        Args:
            dictionary(dict): dict to be tested.

        Returns:
            boolean value indicating whether any value in the 'dictionary' is
                also a dict (meaning that 'dictionary' is nested).
        """
        return any(isinstance(d, dict) for d in dictionary.values())

    @staticmethod
    def listify(variable):
        """Stores passed variable as a list (if not already a list).

        Args:
            variable(str or list): variable to be transformed into a list to
                allow iteration.

        Returns:
            variable(list): either the original list, a string converted to a
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
            variable(list): variable to be transformed into a string.

        Returns:
            variable(str): either the original str, a string pulled from a
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
        """Loads object from file into subclass attribute.

        Args:
            name(str): name of attribute for the file contents to be stored.
            file_path(str): a complete file path for the file to be loaded.
            folder(str): a path to the folder where the file should be loaded
                from (not used if file_path is passed).
            file_name(str): contains the name of the file to be loaded without
                the file extension (not used if file_path is passed).
            file_format(str): name of file format in Depot.extensions.

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

        Args:
            variable(any): a python object or a string corresponding to a
                subclass attribute which should be saved to disc.
            file_path(str): a complete file path for the file to be saved.
            folder(str): a path to the folder where the file should be saved
                (not used if file_path is passed).
            file_name(str): contains the name of the file to be saved without
                the file extension (not used if file_path is passed).
            file_format(str): name of file format in Depot.extensions.
        """
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

        A dict called 'options' should be defined here for subclasses to use
        much of the functionality of SimpleClass.

        Generally, the 'checks' attribute should be set here if the subclass
        wants to make use of related methods.
        """

        return self

    def edit(self, keys = None, values = None, options = None):
        """Updates 'options' dictionary with passed arguments.

        Args:
            keys(str or list): a string name or list of names for keys in the
                'options' dict.
            values(object or list(object)): siMpLify compatible objects which
                can be integrated in the package framework. If they are custom
                algorithms, they should be subclassed from SimpleTechnique to
                ensure compatibility.
            options(dict): a dictionary with keys of techniques and values of
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
    def publish(self, **kwargs):
        """Required method which creates any objects to be applied to data or
        variables.

        In the case of iterative classes, such as Cookbook, this method should
        construct any plans to be later implemented by the 'implement' method.

        Args:
            **kwargs: keyword arguments are not ordinarily included in the
                publish method. But nothing precludes them from being added
                to subclasses.
        """
        # Runs attribute checks from list in 'checks' attribute (if it exists).
        self._run_checks()
        return self


@dataclass
class Stage(SimpleClass):

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

    """ Public Methods """

    def change(self, new_state):
        """Changes 'state' to 'new_state'.

        Args:
            new_state(str): name of new state matching a string in 'states'.

        Raises:
            TypeError: if new_state is not in 'states'.

        """
        if new_state in self.states:
            self.state = new_state
        else:
            error = new_state + ' is not a recognized data state'
            raise TypeError(error)

    """ Core siMpLify Methods """

    def draft(self):
        # Sets possible states
        self.states = []
        for stage in self.idea.options['simplify']['packages']:
            if stage in self.idea.options and stage == 'farmer':
                for step in self.idea.options['farmer']['farmer_techniques']:
                    self.states.append(step)
            else:
                self.states.append(stage)
        self.state = self.states[0]
        return self


@dataclass
class SimpleChecker(SimpleClass):

    def __post_init__(self):
        self.draft()
        return self

    """ Private Methods """

    def _check_depot(self, instance):
        """Adds a Depot instance with default settings as 'depot' attribute if
        one was not passed when the subclass was instanced.

        Args:
            instance (SimpleClass): subclass to be checked.

        Raises:
            TypeError: if 'depot' is neither a str, None, or Depot instance.

        """
        # Local import to avoid circular dependency.
        from simplify import Depot
        if instance.exists('depot'):
            if isinstance(instance.depot, str):
                instance.depot = Depot(root_folder = instance.depot)
            elif isinstance(instance.depot, Depot):
                pass
            else:
                error = 'depot must be a string type or Depot instance'
                raise TypeError(error)
        else:
            instance.depot = Depot()
        if not hasattr(SimpleClass, 'depot'):
            setattr(SimpleClass, 'depot', instance.depot)
        return instance

    def _check_gpu(self, instance):
        """If gpu status is not set, checks if the local machine has a GPU
        capable of supporting included machine learning algorithms.

        Because the tensorflow 'is_gpu_available' method is very lenient in
        counting what qualifies, it is recommended to set the 'gpu' attribute
        directly or through an Idea instance.

        Args:
            instance (SimpleClass): subclass to be checked.

        """
#        if hasattr(self, 'gpu'):
#            if instance.gpu and instance.verbose:
#                print('Using GPU')
#            elif instance.verbose:
#                print('Using CPU')
#        elif is_gpu_available:
#            instance.gpu = True
#            if instance.verbose:
#                print('Using GPU')
#        else:
#            instance.gpu = False
#            if instance.verbose:
#                print('Using CPU')
        return instance

    def _check_ingredients(self, instance, ingredients = None):
        """Checks if ingredients attribute exists and takes appropriate action.

        If an 'ingredients' attribute exists, it determines if it contains a
        file folder, file path, or Ingredients instance. Depending upon its
        type, different actions are taken to actually create an Ingredients
        instance.

        If ingredients is None, then an Ingredients instance is
        created with no pandas DataFrames or Series within it.

        Args:
            instance (SimpleClass): subclass to be checked.

            ingredients (Ingredients, a file path containing a DataFrame or
                Series to add to an Ingredients instance, a folder
                containing files to be used to compose Ingredients DataFrames
                and/or Series, DataFrame, Series, or numpy array).

        Raises:
            TypeError: if 'ingredients' is neither a str, None, DataFrame,
                Series, numpy array, or Ingredients instance.

        """
        # Local import to avoid circular dependency.
        from simplify import Ingredients
        # Assigns passed 'ingredients' to 'ingredients' attribute
        if ingredients is not None:
            instance.ingredients = ingredients
        # If 'ingredients' is a data container, it is assigned to 'df' in a new
        # instance of Ingredients assigned to 'ingredients'.
        # If 'ingredients' is a file path, the file is loaded into a DataFrame
        # and assigned to 'df' in a new Ingredients instance at 'ingredients'.
        # If 'ingredients' is None, a new Ingredients instance is created and
        # assigned to 'ingreidents' with no attached DataFrames.
        if (isinstance(instance.ingredients, pd.Series)
                or isinstance(instance.ingredients, pd.DataFrame)
                or isinstance(instance.ingredients, np.ndarray)):
            instance.ingredients = Ingredients(df = instance.ingredients)
        elif isinstance(instance.ingredients, str):
            if os.path.isfile(instance.ingredients):
                df = instance.depot.load(
                    folder = instance.depot.data,
                    file_name = instance.ingredients)
                instance.ingredients = Ingredients(df = df)
            elif os.path.isdir(instance.ingredients):
                instance.depot.create_glob(folder = instance.ingredients)
                instance.ingredients = Ingredients()
        elif instance.ingredients is None:
            instance.ingredients = Ingredients()
        else:
            error = 'ingredients must be a string, DataFrame, or an instance'
            raise TypeError(error)
        return instance

    def _check_name(self, instance):
        """Sets 'name' attribute if one does not exist in subclass.

        A separate 'name' attribute is used throughout the package so that
        users can set their own naming conventions or use the names of parent
        classes when subclassing without being dependent upon
        __class__.__name__.

        If no 'name' attribute exists (usually defined in the 'draft' method),
        then __class__.__name__ is used as the default backup.

        """
        if not instance.exists('name'):
            instance.name = instance.__class__.__name__.lower()
        return instance

    """ Core siMpLify Methods """

    def draft(self):
        self.checks = []
        super().draft()
        return self

    def publish(self, instance, checks = None):
        """Checks attributes from 'checks' and runs corresponding methods based
        upon strings stored in 'checks'.

        Those methods should have the prefix '_check_' followed by the string
        in the attribute 'checks'. Any subclass seeking to add new checks can
        add a new method using those naming conventions with the only passed
        argument as 'self'.

        Args:
            instance(SimpleClass): instance to have checks run against.
            checks(list or str): list or name of checks to be run. If not
                passed, the list in the 'checks' attribute with be used.

        Returns:
            instance(SimpleClass): instance will modifications made based upon
                checks performed.
        Raises:
            AttributeError: if correspondig method name is neither in the
                passed instance nor this class instance.
        """
        if not checks and hasattr(instance, 'checks'):
            checks = instance.checks
        if checks:
            for check in self.listify(checks):
                try:
                    getattr(instance, '_check_' + check)()
                except AttributeError:
                    try:
                        instance = getattr(self, '_check_' + check)(
                            instance = instance)
                    except AttributeError:
                        error = check + ' does not correspond to method name'
                        raise AttributeError(error)
        return instance