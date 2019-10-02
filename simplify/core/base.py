"""
.. module:: base
:synopsis: core parent classes of siMpLify package
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0

Contents:
    SimpleClass: parent abstract base class for all siMpLify classes.
    SimpleManager: parent class for the builder classes.
    SimplePlan: parent container class for storing iterables created by
        SimpleManager subclasses.
    SimpleStep: parent class of the iterable steps of the SimplePlan
        subclasses.
    SimpleTechnique: parent class of algorithms used by SimpleStep subclasses.
    Simplify: controller class for highly-automated siMpLify projects.

This module contains the parent classes used by the siMpLify package and
should be subclassed in any additional extensions to the siMpLify package.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from itertools import product
import os
import warnings

from more_itertools import unique_everseen
import numpy as np
import pandas as pd
#from tensorflow.test import is_gpu_available

from simplify.core.decorators import localize


@dataclass
class SimpleClass(ABC):
    """Absract base class for classes in siMpLify package to support a common
    architecture and allow for sharing of universal methods.

    SimpleClass creates a code structure patterned after the writing process.
    It divides processes into four stages which are the names or prefixes to
    the core methods used throughout the siMpLify package:

        1) draft: sets default attributes (required).
        2) edit: makes any desired changes to the default attributes.
        3) publish: creates python objects based upon those attributes.
        4) produce: applies those publishd objects to passed variables
            (usually data).

    If the subclass includes boolean attributes of 'auto_publish' or
    'auto_produce', and those attributes are set to True, then the 'publish'
    and/or 'produce' methods are called when the class is instanced.

    Args:
        These arguments are not required for any SimpleClass, but are commonly
        used throughout the package. Brief descriptions are included here:

        idea(Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located. Once an
            Idea instance is created by a subclass of SimpleClass, it is
            automatically made available to all other SimpleClass subclasses
            that are instanced in the future.
        depot(Depot): an instance of Depot. Once a Depot instance is created by
            a subclass of SimpleClass, it is automatically made available to all
            other SimpleClass subclasses that are instanced in the future.
        ingredients(Ingredients or str): an instance of Ingredients of a string
            containing the full file path of where a supported file type that
            can be loaded into a pandas DataFrame is located. If it is a string,
            the loaded DataFrame will be bound to a new ingredients instance as
            the 'df' attribute.
        name(str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced.
        auto_produce(bool): whether to call the 'produce' method when the class
            is instanced.

    """

    def __post_init__(self):
        """Calls selected initialization methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Creates 'idea' attribute if a string is passed to Idea when subclass
        # was instanced. Injects attributes from 'idea' to subclass.
        if self.__class__.__name__ != 'Idea':
            self._check_idea()
            # Sets appropriate state of siMpLify package using 'state_machine' 
            # created by Idea instance.    
            self.state_machine.advance()
        # Calls draft method to set up class instance defaults.
        self.draft()
        # Runs attribute checks from list in 'checks' attribute (if it exists).
        self._run_checks()
        # Converts values in 'options' to classes by lazily importing them.
        self._lazily_import_options()
        # Registers subclass into lists based upon specific subclass needs.
        self._register_subclass()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
            # Calls 'produce' method if 'auto_produce' is True.
            if hasattr(self, 'auto_produce') and self.auto_produce:
                self.produce()
        return self

    """ Magic Methods """

    def __call__(self, idea, *args, **kwargs):
        """When called as a function, a subclass will return the produce method
        after running __post_init__.

        Args:
            idea(Idea or str): an instance of Idea or path where an Idea
                configuration file is located. This argument must be passed when
                a subclass is called as a function.
            *args and **kwargs (any): passed to the 'produce' method.

        Returns:
            return value of 'produce' method.

        """
        self.idea = idea
        self.auto_publish = True
        self.auto_produce = False
        self.__post_init__()
        return self.produce(*args, **kwargs)

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
        if item in self.options:
            del self.options[item]
        elif self.exists(item):
            setattr(self, item, None)
        else:
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
        # Intercepts common dict methods and applies them to 'options' dict.
        if attr in ['clear', 'items', 'pop', 'keys', 'update', 'values']:
            return getattr(self.options, attr)
        elif attr in self.__dict__:
            return self.__dict__[attr]
        elif attr.startswith('__') and attr.endswith('__'):
            error = 'Access to magic methods not permitted through __getattr__'
            raise AttributeError(error)
        else:
            error = attr + ' not found in ' + self.__class__.__name__
            raise AttributeError(error)

    def __getitem__(self, item):
        """Returns item if 'item' is in 'options' or is an atttribute.

        Args:
            item(str): item matching 'options' dictionary key or attribute name.

        Returns:
            Value for item in 'options', 'item' attribute value, or None if
                neither of those exist.

        """
        if item in self.options:
            return self.options[item]
        elif hasattr(self, item):
            return getattr(self, item)
        else:
            return None

    def __iter__(self):
        """Returns 'options' to mirror dictionary functionality."""
        return self.options

    def __setitem__(self, item, value):
        """Adds item and value to options dictionary.

        Args:
            item(str): 'options' key to be set.
            value(any): corresponding value to be set for 'item' key in
                'options'.

        """
        self.options[item] = value
        return self

    """ Private Methods """

    def _check_depot(self):
        """Adds a Depot instance with default settings as 'depot' attribute if
        one was not passed when the subclass was instanced.
        """
        # Local import to avoid circular dependency.
        from simplify import Depot
        if self.exists('depot'):
            if isinstance(self.depot, str):
                self.depot = Depot(root_folder = self.depot)
        else:
            self.depot = Depot()
        return self

    def _check_gpu(self):
        """If gpu status is not set, checks if the local machine has a GPU
        capable of supporting included machine learning algorithms.

        Because the tensorflow 'is_gpu_available' method is very lenient in
        counting what qualifies, it is recommended to set the 'gpu' attribute
        directly or through an Idea instance.
        """
#        if hasattr(self, 'gpu'):
#            if self.gpu and self.verbose:
#                print('Using GPU')
#            elif self.verbose:
#                print('Using CPU')
#        elif is_gpu_available:
#            self.gpu = True
#            if self.verbose:
#                print('Using GPU')
#        else:
#            self.gpu = False
#            if self.verbose:
#                print('Using CPU')
        return self

    def _check_idea(self):
        """Loads Idea settings from a file if a string is passed instead of an
        Idea instance.

        Injects sections of 'idea' to a subclass instance using
        user settings.
        """
        # Local import to avoid circular dependency.
        from simplify import Idea
        if self.exists('idea') and isinstance(self.idea, str):
            self.idea = Idea(configuration = self.idea)
        # Adds attributes to class from appropriate sections of the idea.
        sections = ['general']
        if self.exists('idea_sections'):
            if isinstance(self.idea_sections, str):
                sections.append(self.idea_sections)
            elif isinstance(self.idea_sections, list):
                sections.extend(self.idea_sections)
        if (hasattr(self, 'name')
                and self.name in self.idea.configuration
                and not self.name in sections):
            sections.append(self.name)
        self = self.idea.inject(instance = self, sections = sections)
        return self

    def _check_ingredients(self, ingredients = None):
        """Checks if ingredients attribute exists and takes appropriate action.

        If an 'ingredients' attribute exists, it determines if it contains a
        file folder, file path, or Ingredients instance. Depending upon its
        type, different actions are taken to actually create an Ingredients
        instance.

        If ingredients is None, then an Ingredients instance is
        created with no pandas DataFrames or Series within it.

        Args:
            ingredients (Ingredients, a file path containing a DataFrame or
                Series to add to an Ingredients instance, or a folder
                containing files to be used to compose Ingredients DataFrames
                and/or Series).
        """
        # Local import to avoid circular dependency.
        from simplify import Ingredients
        if ingredients:
            self.ingredients = ingredients
        if (isinstance(self.ingredients, pd.Series)
                or isinstance(self.ingredients, pd.DataFrame)):
            self.ingredients = Ingredients(df = self.ingredients)
        elif isinstance(self.ingredients, str):
            if os.path.isfile(self.ingredients):
                df = self.depot.load(folder = self.depot.data,
                                     file_name = self.ingredients)
                self.ingredients = Ingredients(df = df)
            elif os.path.isdir(self.ingredients):
                self.depot.create_glob(folder = self.ingredients)
                self.ingredients = Ingredients()
        elif not self.ingredients:
            self.ingredients = Ingredients()
        return self

    def _check_steps(self):
        """Creates 'steps' attribute from one of several sources.

        Initially, the method checks for 'steps' attribute. If that doesn't
        exist, it searches for an attribute with the class 'name' prefix
        followed by 'steps' (e.g. 'cookbook_steps'). This latter attribute is
        often created from the Idea instance.

        If the value stored in the 'steps' attribute is 'all', 'none', or
        'default', the steps value is changed using the '_convert_wildcards'
        method.

        Also, a 'step' attribute is created from the first item in the 'steps'
        attribute.
        """
        if not self.exists('steps'):
            if hasattr(self, self.name + '_steps'):
                self.steps = self._convert_wildcards(getattr(
                        self, self.name + '_steps'))
        else:
            self.steps = self.listify(self.steps)
        if not self.exists('step'):
            self.step = self.steps[0]
        return self

    def _convert_wildcards(self, value):
        """Converts 'all', 'default', or 'none' values to a list of items.

        Args:
            value (list or str): name(s) of techniques, steps, or managers.

        Returns:
            If 'all', all keys listed in 'options' dictionary are returned.
            If 'default', 'default_operations' are returned or, if they don't
                exist, all keys listed in 'options' dictionary are returned.
            Otherwise, 'techniques' is returned intact.
        """
        if value in ['all', ['all']]:
            return list(self.options.keys())
        elif value in ['default', ['default']]:
            if (hasattr(self, 'default_operations')
                    and self.default_operations):
                return self.default_operations
            else:
                return list(self.options.keys())
        elif value in ['none', ['none'], None]:
            return 'none'
        else:
            return value

    def _lazily_import_options(self):
        """Limits module imports to only needed package dependencies.

        This method allows users to either save memory or have less
        dependencies locally available by importing fewer packages than
        would be done through normal, blanket importation.

        To use this method, 'options' should be formatted as follows:
            {name(str): [module_path(str), class_name(str)]}

        """
        imported_options = {}
        if self.exists('options'):
            if not hasattr(self, 'lazy_import') or self.lazy_import:
                if self.has_list_values(self.options):
                    for name, settings in self.options.items():
                        imported_options.update(
                            {name: getattr(import_module(settings[0]),
                                           settings[1])})
                    self.options = imported_options
        return self

    @classmethod
    def _register_subclass(self):
        """Adds subclass to appropriate list based on attribute in subclass."""
        self._registered_subclasses = {
            'registered_state_subclasses': 'state_dependent'}
        for subclass_list, attribute in self._registered_subclasses.items():
            if not hasattr(self, subclass_list):
                setattr(self, subclass_list, [])
            if hasattr(self, attribute) and getattr(self, attribute):
                getattr(self, subclass_list).append(self)
        return self

    def _run_checks(self):
        """Checks attributes from 'checks' and runs corresponding methods based
        upon strings stored in 'checks'.

        Those methods should have the prefix '_check_' followed by the string
        in the attribute 'checks' and have no parameters other than 'self'. Any
        subclass seeking to add new checks can add a new method using those
        naming conventions.
        """
        if self.exists('checks'):
            for check in self.checks:
                getattr(self, '_check_' + check)()
        return self

    """ Public Tool Methods """

    def conform(self, step):
        """Sets 'step' attribute to passed 'step' throughout package.

        This method is used to maintain a universal state in the package for
        subclasses that are state dependent. It iterates through any subclasses
        listed in 'registered_state_subclasses' to call their 'conform'
        methods.

        Args:
            step(str): corresponds to current state in siMpLify package.
        """
        self.step = step
        for _subclass in self.registered_state_subclasses:
            _subclass.conform(step = step)
        return self

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
        if isinstance(iterable, list):
            return list(unique_everseen(iterable))
        elif isinstance(iterable, pd.Series):
            return iterable.drop_duplicates(inplace = True)
        elif isinstance(iterable, pd.DataFrame):
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
        if isinstance(values, list) and ignore_values_list:
            return dict.fromkeys(keys, values)
        else:
            return dict(zip(keys, values))

    def exists(self, attribute):
        """Returns if attribute exists in subclass and is not None.

        Args:
            attribute (str): name of attribute to be evaluated.

        Returns:
            boolean value indicating whether the attribute exists and is not
                None.
        """
        return (hasattr(self, attribute)
                and getattr(self, attribute) is not None)

    @staticmethod
    def has_list_values(dictionary):
        """Returns if passed 'dictionary' has lists for values.

        Args:
            dictionary (dict): dict to be tested.

        Returns:
            boolean value indicating whether any value in the 'dictionary' has
                list for all values
        """
        return all(isinstance(d, list) for d in dictionary.values())

    @staticmethod
    def is_nested(dictionary):
        """Returns if passed 'dictionary' is nested at least one-level.

        Args:
            dictionary (dict): dict to be tested.

        Returns:
            boolean value indicating whether any value in the 'dictionary' is
                also a dict (meaning that 'dictionary' is nested).
        """
        return any(isinstance(d, dict) for d in dictionary.values())

    @staticmethod
    def listify(variable):
        """Stores passed variable as a list (if not already a list).

        Args:
            variable (str or list): variable to be transformed into a list to
                allow iteration.

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

#    @staticmethod
#    def nestify(keys, dictionaries):
#        """Converts dict to nested dict if not already one.
#
#        Args:
#            keys (list): list to be keys to nested dict.
#            dictionaries (dict, list(dicts)): dictionary (or list of same) to
#                be values in nested dictionary.
#
#        Returns:
#            nested dict with keys added to outer layer, with either the same
#            dict as all values (if one dict) or each dict in the list as a
#            corresponding value to each key, original dict (if already nested),
#            or an empty dict.
#        """
#        if isinstance(dictionaries, list):
#            return dict(zip(keys, dictionaries))
#        elif (isinstance(dictionaries, dict)
#                and any(isinstance(i, dict) for i in dictionaries.values())):
#            return dictionaries
#        elif isinstance(keys, list) and dictionaries is not None:
#            return dict.fromkeys(keys, dictionaries)
#        else:
#            return {}

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
        """Loads object from file into subclass attribute.

        Args:
            name: name of attribute for the file contents to be stored.
            file_path: a complete file path for the file to be loaded.
            folder: a path to the folder where the file should be loaded from
                (not used if file_path is passed).
            file_name: a string containing the name of the file to be loaded
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                Depot.extensions.
        """
        setattr(self, name, self.depot.load(file_path = file_path,
                                            folder = folder,
                                            file_name = file_name,
                                            file_format = file_format))
        return self

    def save(self, variable = None, file_path = None, folder = None,
             file_name = None, file_format = None):
        """Exports a variable or attribute to disc.

        Args:
            variable: a python object or a string corresponding to a subclass
                attribute which should be saved to disc.
            file_path: a complete file path for the file to be saved.
            folder: a path to the folder where the file should be saved (not
                used if file_path is passed).
            file_name: a string containing the name of the file to be saved
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                Depot.extensions.
        """
        if isinstance(variable, str):
            variable = getattr(self, variable)
        self.depot.save(variable = variable,
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
        pass
        return self

    def edit(self, techniques = None, algorithms = None, options = None):
        """Updates 'options' dictionary with passed arguments.

        Args:
            techniques (str or list): a string name or list of names for keys
                in the 'options' dict.
            algorithms (object or list(object)): siMpLify compatible objects
                which can be integrated in the package framework. If they are
                custom algorithms, they should be subclassed from SimpleStep to
                ensure compatibility.
            options (dict): a dictionary with keys of techniques and values of
                algorithms. This should be passed if the user has already
                combined some or all 'techniques' and 'algorithms' into a dict.
        """
        if not self.exists('options'):
            self.options = {}
        if options:
            self.options.update(options)
        if techniques and algorithms:
            self.options.update(dict(zip(techniques, algorithms)))
        return self

    @abstractmethod
    def publish(self, **kwargs):
        """Required method which creates any objects to be applied to data or
        variables.

        In the case of iterative classes, such as Cookbook, this method should
        construct any plans to be later implemented by the 'produce' method.

        Args:
            **kwargs: keyword arguments are not ordinarily included in the
                publish method. But nothing precludes them from being added
                to subclasses.
        """
        pass
        return self


@dataclass
class SimpleManager(SimpleClass):
    """Parent class for siMpLify planners like Cookbook, Almanac, Analysis,
    and Canvas.

    This class adds methods useful to create iterators, iterate over user
    options, and transform data or fit models.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.
    """

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _check_plan_iterable(self):
        """Creates plan iterable attribute to be filled with concrete plans if
        one does not exist."""
        if not self.exists('plan_iterable'):
            self.plan_iterable = 'plans'
            self.plans = {}
        elif not self.exists(self.plan_iterable):
            setattr(self, self.plan_iterable, {})
        return self

    def _create_steps_lists(self):
        """Creates list of lists of all possible steps in 'options'."""
        self.all_steps = []
        for step in self.options.keys():
            # Stores each step attribute in a list.
            if hasattr(self, step):
                setattr(self, step, self.listify(getattr(self, step)))
            # Stores a list of 'none' if there is no corresponding local
            # attribute.
            else:
                setattr(self, step, ['none'])
            # Adds step to a list of all steps.
            self.all_steps.append(getattr(self, step))
        return self

    def _publish_plans_parallel(self):
        """Creates plan iterable from list of lists in 'all_steps'."""
        # Creates a list of all possible permutations of step lists.
        all_plans = list(map(list, product(*self.all_steps)))
        for i, plan in enumerate(all_plans):
            publishd_steps = {}
            for j, (step_name, step_class) in enumerate(self.options.items()):
                publishd_steps.update(
                        {step_name: step_class(technique = plan[j])})
            getattr(self, self.plan_iterable).update(
                    {i + 1: self.plan_class(steps = publishd_steps,
                                             number = i + 1)})
        return self

    def _publish_plans_serial(self):
        """Creates plan iterable from list of lists in 'all_steps'."""
        for i, (plan_name, plan_class) in enumerate(self.options.items()):
            for steps in self.all_steps[i]:
                getattr(self, self.plan_iterable).update(
                        {i + 1: plan_class(steps = steps)})
        return self

    def _produce_parallel(self, variable = None, **kwargs):
        """Method that implements all of the publishd objects on the
        passed variable.

        The variable is returned after being transformed by called methods.

        Args:
            variable(any): any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        for number, plan in getattr(self, self.plan_iterable).items():
            if self.verbose:
                print('Testing', self.name, str(number))
            variable = plan.produce(variable, **kwargs)
        return variable

    def _produce_serial(self, variable = None, **kwargs):
        """Method that implements all of the publishd objects on the
        passed variable.

        The variable is returned after being transformed by called methods.

        Args:
            variable(any): any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        for number, plan in getattr(self, self.plan_iterable).items():
            variable = plan.produce(variable, **kwargs)
        return variable

    """ Core siMpLify methods """

    def draft(self):
        """ Declares defaults for class."""
        self.options = {}
        self.checks = ['steps', 'depot', 'plan_iterable']
        self.state_attributes = ['depot', 'ingredients']
        return self

    def publish(self):
        """Finalizes iterable dict of plans with instanced plan classes."""
        self._create_steps_lists()
        getattr(self, '_publish_plans_' + self.manager_type)()
        return self

    def produce(self, variable = None, **kwargs):
        """Method that implements all of the publishd objects on the
        passed variable.

        The variable is returned after being transformed by called methods.

        Args:
            variable(any): any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        variable = getattr(self, '_produce_' + self.manager_type)(
                variable = variable, **kwargs)
        return variable


@dataclass
class SimplePlan(SimpleClass):
    """Class for containing plan classes like Recipe, Harvest, Review, and
    Illustration.

    Args:
        steps(dict): dictionary containing keys of step names (strings) and
            values of SimpleStep subclass instances.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.
    """

    steps: object = None

    def __post_init__(self):
        # Adds name of SimpleManager subclass to sections to inject from Idea
        # so that all of those section entries are available as local
        # attributes.
        if self.exists('manager_name'):
            self.idea_sections = [self.manager_name]
        super().__post_init__()
        return self

    def __call__(self, *args, **kwargs):
        """When called as a function, a SimplePlan class or subclass instance
        will return the 'produce' method.
        """
        return self.produce(*args, **kwargs)

    def draft(self):
        """SimplePlan's generic 'draft' method."""
        pass
        return self

    def publish(self):
        """SimplePlan's generic 'publish' method requires no extra
        preparation.
        """
        pass
        return self

    def produce(self, variable, **kwargs):
        """Iterates through SimpleStep techniques 'produce' methods.

        Args:
            variable(any): variable to be changed by serial SimpleManager
                subclass.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        # If 'data_variable' is not set, attempts to infer its name from passed
        # variable.
        if not self.data_variable and hasattr(variable, 'name'):
            self.data_variable = variable.name
        for step, technique in self.steps.items():
            setattr(self, self.data_variable, technique.produce(
                    getattr(self, self.data_variable), **kwargs))
        return self


@dataclass
class SimpleStep(SimpleClass):
    """Parent class for various steps in the siMpLify package.

    SimpleStep, unlike the above subclasses of SimpleClass, should have a
    'parameters' parameter as an attribute to the class instance for the
    included methods to work properly. Otherwise, 'parameters' will be set to
    an empty dict.

    'fit', 'fit_transform', and 'transform' adapter methods are included in
    SimpleClass to support partial scikit-learn compatibility.

    Args:
        techniques(list of str): name of technique(s) that match(es) string(s)
            in the 'options' keys or a wildcard value such as 'default', 'all',
            or 'none'.
        parameters(dict): parameters to be attached to algorithm in 'options'
            corresponding to 'techniques'. This parameter need not be passed to
            the SimpleStep subclass if the parameters are in the accessible
            Idea instance or if the user wishes to use default parameters.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.
    """

    technique: object = None
    parameters: object = None
    auto_publish: bool = True

    def __post_init__(self):
        # Adds name of SimpleManager subclass to sections to inject from Idea
        # so that all of those section entries are available as local
        # attributes.
        if self.exists('manager_name'):
            self.idea_sections = [self.manager_name]
        super().__post_init__()
        return self

    """ Private Methods """

    def _check_parameters(self):
        """Adds empty 'parameters' dict if it doesn't exist."""
        if not self.exists('parameters'):
            self.parameters = {}
        return self


    def _denestify(self, technique, parameters):
        """Removes outer layer of 'parameters' dict, if it exists, by using
        'technique' as the key.

        If 'parameters' is not nested, 'parameters' is returned unaltered.
        """
        if self.is_nested(parameters) and technique in parameters:
            return parameters[technique]
        else:
            return parameters

    def _publish_parameters(self):
        """Compiles appropriate parameters for all 'technique'.

        After testing several sources for parameters using '_get_parameters',
        parameters are subselected, if necessary, using '_select_parameters'.
        If 'runtime_parameters' and/or 'extra_parameters' exist in the
        subclass, those are added to 'parameters' as well.
        """
        parameter_groups = ['', '_selected', '_runtime', '_extra',
                            '_conditional']
        for parameter_group in parameter_groups:
            if hasattr(self, '_get_parameters' + parameter_group):
                getattr(self, '_get_parameters' + parameter_group)(
                        technique = self.technique,
                        parameters = self.parameters)

        return self

    def _get_parameters(self, technique, parameters):
        """Returns parameters from different possible sources based upon passed
        'technique'.

        If 'parameters' attribute is None, the accessible Idea instance is
        checked for a section matching the 'name' attribute of the class
        instance. If no Idea parameters exist, 'default_parameters' are used.
        If there are no 'default_parameters', an empty dictionary is created
        for parameters.

        Args:
            technique(str): name of technique for which parameters are sought.

        """
        if self.exists('parameters') and self.parameters:
            self.parameters = self._denestify(self.technique, self.parameters)
        elif self.technique in self.idea.configuration:
            self.parameters = self.idea.configuration[self.technique]
        elif self.name in self.idea.configuration:
            self.parameters = self.idea.configuration[self.name]
        elif self.exists('default_parameters'):
            self.parameters = self._denestify(
                    technique = self.technique,
                    parameters = self.default_parameters)
        else:
            self.parameters = {}
        return self

    def _get_parameters_extra(self, technique, parameters):
        """Adds parameters from 'extra_parameters' if attribute exists.

        Some parameters are stored in 'extra_parameters' because of the way
        the particular algorithms are constructed by dependency packages. For
        example, scikit-learn consolidates all its support vector machine
        classifiers into a single class (SVC). To pick different kernels for
        that class, a parameter ('kernel') is used. Since siMpLify wants to
        allow users to compare different SVC kernel models (linear, sigmoid,
        etc.), the 'extra_parameters attribute is used to add the 'kernel'
        and 'probability' paramters in the Classifier subclass.

        Args:
            technique (str): name of technique selected.
            parameters (dict): a set of parameters for an algorithm.

        Returns:
            parameters (dict) with 'extra_parameters' added if that attribute
                exists in the subclass and the technique is listed as a key
                in the nested 'extra_parameters' dictionary.
        """
        if self.exists('extra_parameters') and self.extra_parameters:
            parameters.update(
                    self._denestify(technique = technique,
                                    parameters = self.extra_parameters))
            return

    def _get_parameters_runtime(self, technique, parameters):
        """Adds runtime parameters to parameters based upon passed 'technique'.

        Args:
            technique(str): name of technique for which runtime parameters are
                sought.
        """
        if self.exists('runtime_parameters'):
            parameters.update(
                    self._denestify(technique = technique,
                                    parameters = self.runtime_parameters))
            return parameters


    def _get_parameters_selected(self, technique, parameters,
                                 parameters_to_use = None):
        """For subclasses that only need a subset of the parameters stored in
        idea, this function selects that subset.

        Args:
            parameters_to_use(list or str): list or string containing names of
                parameters to include in final parameters dict.
        """
        if self.exists('selected_parameters') and self.selected_parameters:
            if not parameters_to_use:
                if isinstance(self.selected_parameters, list):
                    parameters_to_use = self.selected_parameters
                elif self.exists('default_parameters'):
                    parameters_to_use = list(self._denestify(
                            technique, self.default_parameters).keys())
            new_parameters = {}
            for key, value in parameters.items():
                if key in self.listify(parameters_to_use):
                    new_parameters.update({key: value})
            self.parameters = new_parameters
        return self

    """ Core siMpLify Public Methods """

    def draft(self):
        """Default draft method which sets bare minimum requirements.

        This default draft should only be used if users are planning to
        manually add all options and parameters to the SimpleStep subclass.
        """
        self.options = {}
        self.checks = ['idea', 'parameters']
        return self

    def edit_parameters(self, technique, parameters):
        """Adds a parameter set to parameters dictionary.

        Args:
            parameters(dict): dictionary of parameters to be added to
                'parameters' of subclass.

        Raises:
            TypeError: if 'parameters' is not dict type.
        """
        if isinstance(parameters, dict):
            if not hasattr(self, 'parameters') or self.parameters is None:
                self.parameters = {technique: parameters}
            else:
                self.parameters[technique].update(parameters)
            return self
        else:
            error = 'parameters must be a dict type'
            raise TypeError(error)

    def publish(self):
        """Finalizes parameters and adds 'parameters' to 'algorithm'."""
        self.technique = self._convert_wildcards(value = self.technique)
        self._publish_parameters()
        if self.technique in ['none', 'None', None]:
            self.technique = 'none'
            self.algorithm = None
        elif (self.exists('custom_options')
                and self.technique in self.custom_options):
            self.algorithm = self.options[self.technique](
                    parameters = self.parameters)
        else:
            self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def produce(self, ingredients, plan = None):
        """Generic implementation method for SimpleStep subclass.

        This method should only be used if the algorithm is to be applied to
        'x' and 'y' in ingredients and sklearn compatible 'fit' and 'transform'
        methods are available.

        Args:
            ingredients (Ingredients): an instance of Ingredients or subclass.
            plan (SimplePlan subclass or instance): is not used by the generic
                method but is made available as an optional keyword for
                compatibility with other 'produce'  methods. This parameter is
                used when the current SimpleStep subclass needs to look back at
                previous SimpleSteps (as in Cookbook steps).
        """
        if self.algorithm != 'none':
            self.algorithm.fit(ingredients.x_train, ingredients.y_train)
            ingredients.x_train = self.algorithm.transform(ingredients.x_train)
        return ingredients


    """ Scikit-Learn Compatibility Methods """

    def fit(self, x = None, y = None, ingredients = None):
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x(DataFrame or ndarray): independent variables/features.
            y(DataFrame, Series, or ndarray): dependent variable(s)/feature(s)
            ingredients(Ingredients): instance of Ingredients containing
                x_train and y_train attributes (based upon possible remapping).

        Raises:
            AttributeError if no 'fit' method exists for local 'algorithm'.
        """
        if hasattr(self.algorithm, 'fit'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    self.algorithm.fit(x)
                else:
                    self.algorithm.fit(x, y)
            elif ingredients is not None:
                ingredients = self.algorithm.fit(ingredients.x_train,
                                                 ingredients.y_train)
        else:
            error = 'fit method does not exist for this algorithm'
            raise AttributeError(error)
        return self

    def fit_transform(self, x = None, y = None, ingredients = None):
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x(DataFrame or ndarray): independent variables/features.
            y(DataFrame, Series, or ndarray): dependent variable(s)/feature(s)
            ingredients(Ingredients): instance of Ingredients containing
                x_train and y_train attributes (based upon possible remapping).

        Returns:
            transformed x or ingredients, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or ingredients is not passed to
                the method.
        """
        self.fit(x = x, y = y, ingredients = ingredients)
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            return self.transform(x = x, y = y)
        elif ingredients is not None:
            return self.transform(ingredients = ingredients)
        else:
            error = 'fit_transform requires DataFrame, ndarray, or Ingredients'
            raise TypeError(error)

    def transform(self, x = None, y = None, ingredients = None):
        """Generic transform method for partial compatibility to sklearn.
        Args:
            x(DataFrame or ndarray): independent variables/features.
            y(DataFrame, Series, or ndarray): dependent variable(s)/feature(s)
            ingredients(Ingredients): instance of Ingredients containing
                x_train and y_train attributes (based upon possible remapping).

        Returns:
            transformed x or ingredients, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'algorithm'.
        """
        if hasattr(self.algorithm, 'transform'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    x = self.algorithm.transform(x)
                else:
                    x = self.algorithm.transform(x, y)
                return x
            elif ingredients is not None:
                ingredients = self.algorithm.transform(ingredients.x_train,
                                                       ingredients.y_train)
                return ingredients
        else:
            error = 'transform method does not exist for this algorithm'
            raise AttributeError(error)


@dataclass
class SimpleTechnique(SimpleStep):
    """Parent class for various techniques in the siMpLify package.

    SimpleTechnique is the lowest-level parent class in the siMpLify package.
    It follows the general structure of SimpleClass, but is focused on storing
    and applying single techniques to data or other variables. It is included,
    in part, to achieve the highest level of compatibility with scikit-learn as
    currently possible.

    Not every low-level technique needs to a subclass of SimpleTechnique. For
    example, many of the algorithms used in the Cookbook steps (RandomForest,
    XGBClassifier, etc.) are dependencies that are fully integrated into the
    siMpLify architecture without wrapping them into a SimpleTechnique
    subclass. SimpleTechnique is used for custom techniques and for
    dependencies that require a substantial adapter to integrate into siMpLify.

    SimpleTechnique, similar to SimpleStep, should have a 'parameters'
    parameter as an attribute to the class instance for the included methods to
    work properly. Otherwise, 'parameters' will be set to an empty dict.

    Unlike SimpleManager, SimplePlan, and SimpleStep, SimpleTechnique only
    supports a single 'technique'. This is to maximize compatibility to scikit-
    learn and other pipeline scripts.

    Args:
        parameters (dict): parameters to be attached to algorithm in 'options'
            corresponding to 'technique'. This parameter need not be passed to
            the SimpleStep subclass if the parameters are in the accessible
            Idea instance or if the user wishes to use default parameters.
        auto_publish (bool): whether 'publish' method should be called when
            the  class is instanced. This should generally be set to True.

    It is also a child class of SimpleStep. So, its documentation applies as
    well.
    """
    technique: object = None
    parameters: object = None
    auto_publish: bool = True

    def __post_init__(self):
        # Adds name of SimpleStep subclass to sections to inject from Idea
        # so that all of those section entries are available as local
        # attributes.
        if self.exists('step_name'):
            self.idea_sections = [self.step_name]
        super().__post_init__()
        return self

    """ Core siMpLify Public Methods """

    def publish(self):
        """Finalizes parameters and adds 'parameters' to 'algorithm'."""
        self._publish_parameters()
        if self.technique != ['none']:
            self.algorithm = self.options[self.technique](**self.parameters)
        else:
            self.algorithm = None
        return self

    def produce(self, ingredients, plan = None):
        """Generic implementation method for SimpleTechnique subclass.

        Args:
            ingredients(Ingredients): an instance of Ingredients or subclass.
            plan(SimplePlan subclass or instance): is not used by the generic
                method but is made available as an optional keyword for
                compatibility with other 'produce'  methods. This parameter is
                used when the current SimpleTechnique subclass needs to look
                back at previous SimpleSteps.
        """
        if self.algorithm:
            self.algorithm.fit(ingredients.x_train, ingredients.y_train)
            ingredients.x_train = self.algorithm.transform(ingredients.x_train)
        return ingredients


@dataclass
class Simplify(SimpleClass):
    """Controller class for completely automated projects.

    This class is provided for applications that rely exclusively on Idea
    settings and/or subclass attributes. For a more customized application,
    users can access the subpackages ('farmer', 'chef', 'critic', and 'artist')
    directly.

    Args:
        idea(Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located. Once an
            Idea instance is created by a subclass of SimpleClass, it is
            automatically made available to all other SimpleClass subclasses
            that are instanced in the future.
        ingredients(Ingredients or str): an instance of Ingredients or a string
            containing the file path of where a data file for a pandas
            DataFrame is located.
        depot(Depot or str): an instance of Depot a string containing the full
            path of where the root folder should be located for file output.
            Once a Depot instance is created by a subclass of SimpleClass, it is
            automatically made available to all other SimpleClass subclasses
            that are instanced in the future.
        name(str): name of class used to match settings sections in an Idea
            settings file and other portions of the siMpLify package. This is
            used instead of __class__.__name__ so that subclasses can maintain
            the same string name without altering the formal class name.
        auto_publish(bool): sets whether to automatically call the 'publish'
            method when the class is instanced. If you do not plan to make any
            adjustments beyond the Idea configuration, this option should be
            set to True. If you plan to make such changes, 'publish' should be
            called when those changes are complete.
        auto_produce(bool): sets whether to automatically call the 'produce'
            method when the class is instanced.

    """

    idea: object = None
    ingredients: object = None
    depot: object = None
    name: str = 'simplify'
    auto_publish: bool = True
    auto_produce: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    @localize
    def __call__(self, **kwargs):
        """Calls the class as a function.

        Only keyword arguments are accepted so that they can be properly
        turned into local attributes. Those attributes are then used by the
        various 'produce' methods.

        Args:
            **kwargs(list(Recipe) and/or Ingredients): variables that will
                be turned into localized attributes.
        """
        self.__post_init__()
        self.publish()
        self.produce(**kwargs)
        return self

    """ Private Methods """

    def _artist_produce(self):
        self.getattr(self, 'artist').produce(
                ingredients = self.ingredients,
                recipes = self.recipes)
        return self

    def _chef_produce(self):
        self.ingredients, self.recipes = getattr(self, 'chef').produce(
                ingredients = self.ingredients)
        return self

    def _critic_produce(self):
        self.ingredients = getattr(self, 'critic').produce(
                ingredients = self.ingredients,
                recipes = self.recipes)
        return self

    def _farmer_produce(self):
        self.ingredients = getattr(self, 'farmer').produce(
                ingredients = self.ingredients)
        return self

    """ Core siMpLify Methods """

    def draft(self):
        self.options = {
                'farmer': ['simplify.farmer', 'Almanac'],
                'chef': ['simplify.chef', 'Cookbook'],
                'critic': ['simplify.critic', 'Review'],
                'artist': ['simplify.artist', 'Canvas']}
        self.checks = ['depot', 'ingredients']
        return self

    def publish(self):
        self.steps = {}
        for name, settings in self.options.items():
            print(self.subpackages)
            if name in self.subpackages:
                setattr(self, name, self.options[name]())
                getattr(self, name).publish()
                self.steps.update({name: getattr(self, name)})
        return self

    @localize
    def produce(self, **kwargs):
        for step_name, step_instance in self.steps.items():
            getattr(self, step_name + '_produce_')()
        return self