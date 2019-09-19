"""
.. module:: base
  :synopsis: contains core classes of siMpLify package.
  :author: Corey Rayburn Yung
  :copyright: 2019
  :license: CC-BY-NC-4.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from inspect import getfullargspec
from itertools import product
import os
import warnings

from more_itertools import unique_everseen
import pandas as pd
#from tensorflow.test import is_gpu_available


@dataclass
class SimpleClass(ABC):
    """Absract base class for classes in siMpLify package to support a common
    class structure and allow sharing of universal methods.

    To use the class, a subclass must have the following methods:
        draft: a method which sets the default values for the subclass, and
            usually includes the self.options dictionary. If the subclass calls
            super().__post_init__, the 'draft' method is automatically called.
        finalize: a method which, after the user has set all options in the
            preferred manner, constructs the objects which can parse, modify,
            process, analyze, and/or transform data.

    The following methods are not strictly required but should be used if
    the subclass is transforming data or other variable (as opposed to merely
    containing data or variables):
        produce: method which applies the finalized objects to passed data or
            other variables.

    For consistency, methods in subclasses which seek to alter the 'options'
    dict or set parameters should begin with the 'edit_' prefix.

    If the subclass includes boolean attributes of auto_finalize or
    auto_produce, and those attributes are set to True, then the finalize
    and/or produce methods are called when the class is instanced.
    """

    def __post_init__(self):
        """Calls selected initialization methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Creates 'idea' attribute if a string is passed to Idea when subclass
        # was instanced. Injects attributes from 'idea' to subclass.
        if self.__class__.__name__ != 'Idea':
            self._check_idea()
        # Calls draft method to set up class instance defaults.
        self.draft()
        # Runs attribute checks from list in 'checks' attribute (if it exists).
        self._run_checks()
        # Registers subclass into lists based upon specific subclass needs.
        self._register_subclass()
        # Calls 'finalize' method if it exists and 'auto_finalize' is True.
        if hasattr(self, 'auto_finalize') and self.auto_finalize:
            self.finalize()
            # Calls 'produce' method if it exists and 'auto_produce' is True.
            if hasattr(self, 'auto_produce') and self.auto_produce:
                self.produce()
        return self

    """ Magic Methods """

    def __call__(self, idea, *args, **kwargs):
        """When called as a function, a subclass will return the produce method
        after running __post_init__. Any args and kwargs will be passed to the
        'produce' method.

        Args:
            idea (Idea or str): an instance of Idea or path where an Idea
                configuration file is located must be passed when a subclass is
                called as a function.

        Returns:
            return value of 'produce' method.
        """
        self.idea = idea
        self.auto_finalize = True
        self.auto_produce = False
        self.__post_init__()
        return self.produce(*args, **kwargs)

    def __contains__(self, item):
        """Checks if item is in 'options'.

        Args:
            item (str): item to be searched for in 'options' keys.

        Returns:
            True, if 'item' in 'options' - otherwise False.
        """
        return item in self.options

    def __delitem__(self, item):
        """Deletes item if in 'options' or, if ait is an instance attribute, it
        is assigned a value of None.

        Args:
            item (str): item to be deleted from 'options' or attribute to be
                assigned None.

        Raises:
            KeyError: if item is neither in 'options' or an attribute.
        """
        if item in self.options:
            del self.options[item]
        elif hasattr(self, item):
            setattr(self, item, None)
        else:
            error = item + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
        return self

    def __getattr__(self, attr):
        """Returns dict methods applied to options attribute if those methods
        are sought from the class instance.

        Args:
            attr (str): attribute sought.

        Returns:
            attribute or None, if attribute does not exist.

        Raises:
            AttributeError: if a dunder attribute is sought.
        """
        # Intecepts common dict methods and applies them to 'options' dict.
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
        """Returns item if item is in self.options or is an atttribute.

        Args:
            item (str): item matching dict key or attribute name.

        Returns:
            value for item in 'options', 'item' attribute value, or None if
                neither of those exist.
        """
        if item in self.options:
            return self.options[item]
        elif hasattr(self, item):
            return getattr(self, item)
        else:
            return None

    def __iter__(self):
        """Returns options.items() to mirror dict functionality."""
        return self.options.items()

    def __setitem__(self, item, value):
        """Adds item and value to options dictionary.

        Args:
            item (str): 'options' key to be set.
            value (any): corresponding value to be set for 'item' key in
                'options'.
        """
        self.options[item] = value
        return self

    """ Private Methods """

    def _check_depot(self):
        """Adds an Depot instance with default Depot instance as 'depot'
        attribute if one is not passed when subclass is instanced.
        """
        # Local import to avoid circular dependency.
        from simplify import Depot
        if not hasattr(self, 'depot') or self.depot is None:
            self.depot = Depot(idea = self.idea)
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
        Idea instance. Injects sections of 'idea' to a subclass instance using
        user settings.
        """
        # Local import to avoid circular dependency.
        from simplify import Idea
        if hasattr(self, 'idea') and isinstance(self.idea, str):
            self.idea = Idea(file_path = self.idea)
        # Adds attributes to class from appropriate sections of the idea.
        sections = ['general']
        if hasattr(self, 'idea_sections') and self.idea_sections:
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
            ingredients: an Ingredients instance, a file path containing a
                DataFrame or Series to add to an Ingredients instance, or
                a folder containing files to be used to compose Ingredients
                DataFrames and/or Series.
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
        """Checks for 'steps' attribute or finds an attribute with the class
        'name' prefix followed by 'steps' if 'steps' does not already exist or
        is None.
        """
        if not hasattr(self, 'steps') or self.steps is None:
            if hasattr(self, self.name + '_steps'):
                if getattr(self, self.name + '_steps') in ['all', 'default']:
                    self.steps = list(self.options.keys())
                else:
                    self.steps = self.listify(getattr(
                        self, self.name + '_steps'))
            else:
                self.steps = []
        else:
            self.steps = self.listify(self.steps)
        if not hasattr(self, 'step') or not self.step:
            self.step = self.steps[0]
        return self

    @classmethod
    def _register_subclass(self):
        """Adds subclass to appropriate list based on attribute in subclass."""
        self._registered_subclasses = {
            'registered_state_subclasses' : 'state_dependent'}
        for subclass_list, attribute in self._registered_subclasses.items():
            if not hasattr(self, subclass_list):
                setattr(self, subclass_list, [])
            if hasattr(self, attribute) and getattr(self, attribute):
                getattr(self, subclass_list).append(self)
        return self

    def _run_checks(self):
        """Checks attributes from 'checks' and runs corresponding methods based
        upon strings stored in 'checks'. Those methods should have the prefix
        '_check_' followed by the string in the attribute 'checks' and have
        no parameters other than 'self'.
        """
        if hasattr(self, 'checks') and self.checks:
            for check in self.checks:
                getattr(self, '_check_' + check)()
        return self

    """ Public Tool Methods """

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
            keys(list): keys for new dict.
            values(any): valuse for all keys in the new dict or list of values
                corresponding to list of keys.
            ignore_values_list(bool): if value is a list, but the list should
                be the value for all keys, set to True.

        Returns:
            dict with 'keys' as keys and 'values' as all values or zips two
            lists together to form a dict.
        """
        if isinstance(values, list) and ignore_values_list:
            return dict.fromkeys(keys, values)
        else:
            return dict(zip(keys, values))

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
    def nestify(keys, dictionaries):
        """Converts dict to nested dict if not already one.

        Args:
            keys(list): list to be keys to nested dict.
            dictionaries(dict, list(dicts)): dictionary (or list of same) to
                be values in nested dictionary.

        Returns:
            nested dict with keys added to outer layer, with either the same
            dict as all values (if one dict) or each dict in the list as a
            corresponding value to each key, original dict (if already nested),
            or an empty dict.
        """
        if isinstance(dictionaries, list):
            return dict(zip(keys, dictionaries))
        elif (isinstance(dictionaries, dict)
                and any(isinstance(i, dict) for i in dictionaries.values())):
            return dictionaries
        elif isinstance(keys, list) and dictionaries is not None:
            return dict.fromkeys(keys, dictionaries)
        else:
            return {}

    """ Core Public siMpLify Methods """

    def conform(self, step):
        """Sets 'step' attribute to passed 'step' throughout package.

        This method is used to maintain a universal state in the package for
        subclasses that are state dependent. It iterates through any subclasses
        listed in 'registered_state_subclasses' to call their 'conform' methods.

        Args:
            step(str): corresponds to current state in siMpLify package.
        """
        self.step = step
        for _subclass in self.registered_state_subclasses:
            _subclass.conform(step = step)
        return self

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
            techniques (str or list): a string name or list of names for keys in
            the 'options' dict.
            algorithms (object or list(object)): siMpLify compatible objects
                which can be integrated in the package framework. If they are
                custom algorithms, they should be subclassed from SimpleStep to
                ensure compatibility.
            options (dict): a dictionary with keys of techniques and values of
                algorithms. This should be passed if the user has already
                combined some or all 'techniques' and 'algorithms' into a dict.
        """
        if options:
            self.name_to_type.update(options)
        if techniques and algorithms:
            self.name_to_type.update(dict(zip(techniques, algorithms)))
        return self

    @abstractmethod
    def finalize(self, **kwargs):
        """Required method which creates any objects to be applied to data or
        variables.

        In the case of iterative classes, such as Cookbook, this method should
        construct any plans to be later implemented by the 'produce' method. It
        is roughly equivalent to the scikit-learn fit method.

        Args:
            **kwargs: keyword arguments are not ordinarily included in the
                finalize method. But nothing precludes them from being added
                to subclasses.
        """
        pass
        return self

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


@dataclass
class SimpleManager(SimpleClass, ABC):
    """Parent abstract base class for siMpLify planners like Cookbook, Almanac,
    Analysis, and Canvas.

    This class adds a required 'produce' method and other methods useful to
    siMpLify classes which transform data or fit models.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.
    """
    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _create_technique_lists(self):
        """Creates list of option lists of techniques."""
        self.all_steps = []
        for step in self.options.keys():
            # Stores each step attribute in a list
            if hasattr(self, step):
                setattr(self, step, self.listify(getattr(self, step)))
            else:
                setattr(self, step, ['none'])
            # Adds step to a list of all step lists
            self.all_steps.append(getattr(self, step))
        return self

    def _create_plan_instance(self, plan):
        plan_techniques = {}
        for j, (step, technique) in enumerate(self.options.items()):
            # Stores each step attribute in a dict.
            technique_instance = technique(technique = plan[j])
            plan_techniques.update({step : technique_instance})
        return self.plan_class(techniques = plan_techniques)

    def _create_plan_iterables(self):
        # Uses default name of 'plans' if class doesn't have 'plan_iterable'
        # attribute.
        if not hasattr(self, 'plan_iterable'):
            self.plan_iterable = 'plans'
            self.plans = {}
        # Creates empty dict for 'plan_iterable' attribute if attribute doesn't
        # exist.
        elif (not hasattr(self, self.plan_iterable)
                or getattr(self, self.plan_iterable) is None):
            setattr(self, self.plan_iterable, {})
        return self

    def _finalize_parallel(self):
        """Creates dict with steps as keys and techniques as values for each
        plan in the 'plans' attribute."""
        # Creates a list of all possible permutations of step techniques
        # selected. Each item in the the list is an instance of the plan class.
        self.all_plans = list(map(list, product(*self.all_steps)))
        # Iterates through possible steps and either assigns corresponding
        # technique from 'all_plans'.
        for i, plan in enumerate(self.all_plans):
            plan_instance = self._create_plan_instance(plan = plan)
            plan_instance.number = i + 1
            getattr(self, self.plan_iterable).update({i + 1 : plan_instance})
        return self

    def _finalize_serial(self):
        """Creates dict with steps as keys and techniques as values."""
        self.all_plans = dict(zip(self.options.keys(), self.all_steps))
        # Changes 'plan_iterable' attribute to be instance of 'plan_class' with
        # the 'techniques' parameter being the prepared dict.
        setattr(self, self.plan_iterable, self.plan_class(
            techniques = self.all_plans))
        return self

    """ Public Methods """

    def draft(self):
        """ Declares defaults for class."""
        self.options = {}
        self.checks = ['steps', 'depot', 'ingredients']
        self.state_attributes = ['depot', 'ingredients']
        return self

    def finalize(self):
        """Finalizes"""
        self._create_technique_lists()
        self._create_plan_iterables()
        getattr(self, '_finalize_' + self.manager_type)()
        return self

    @abstractmethod
    def produce(self, variable = None, **kwargs):
        """Required method that implements all of the finalized objects on the
        passed variable. The variable is returned after being transformed by
        called methods. It is roughly equivalent to the scikit-learn transform
        method.

        Args:
            variable: any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        pass
        return variable


@dataclass
class SimplePlan(SimpleClass):
    """Class for containing plan classes like Recipe, Harvest, Review, and
    Illustration.

    Args:
        steps(dict): dictionary containing keys of step names (strings) and
            values of SimpleStep subclass instances.

    It is also a child class of SimpleClass. So, its documentation applies as
    well."""

    steps : object = None

    def __post_init__(self):
        # Adds name of SimpleManager subclass to sections to inject from Idea
        # so that all of those section entries are available as local
        # attributes.
        if hasattr(self, 'manager_name'):
            self.idea_sections = [self.manager_name]
        super().__post_init__()
        return self

    def __call__(self, *args, **kwargs):
        """When called as a function, a SimplePlan class or subclass instance
        will return the 'produce' method.
        """
        return self.produce(*args, **kwargs)

    def draft(self):
        """SimplePlan's generic 'draft' method requires no extra defaults."""
        pass
        return self

    def finalize(self):
        """SimplePlan's generic 'finalize' method requires no extra preparation.
        """
        pass
        return self

    def produce(self, variable):
        setattr(self, variable.name, variable)
        for step, technique in self.steps.items():
            if self.verbose:
                print('Applying', step)
            setattr(self, variable.name, technique.produce(variable))
        return self


@dataclass
class SimpleStep(SimpleClass):
    """Parent class for various steps in the siMpLify package.

    SimpleStep, unlike other subclasses of SimpleClass, should have a
    'parameters' parameter as an attribute to the class instance for the
    included methods to work properly. Otherwise, 'parameters' will be set to
    an empty dict.

    Args:
        technique(str): name of technique that matches a string in the 'options'
            keys or a wildcard value such as 'default', 'all', or 'none'.
        parameters(dict): parameters to be attached to algorithm in 'options'
            corresponding to 'technique'. This parameter need not be passed to
            the SimpleStep subclass if the parameters are in the accessible
            Idea instance or if the user wishes to use default parameters.
        auto_finalize(bool): whether 'finalize' method should be called when the
            class is instanced. This should generally be set to True.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.
    """
    techniques : object = None
    parameters : object = None
    auto_finalize : bool = True
    store_names : bool = False

    def __post_init__(self):
        # Adds name of SimpleManager subclass to sections to inject from Idea
        # so that all of those section entries are available as local
        # attributes.
        if hasattr(self, 'manager_name'):
            self.idea_sections = [self.manager_name]
        self.check_nests = ['default_parameters', 'runtime_parameters',
                            'parameters']
        super().__post_init__()
        self._bind_name_decorator()
        return self

    """ Decorators """

    def store_column_names(method):
        """Decorator which creates a complete column list from kwargs passed
        to wrapped method.

        Args:
            method(method): wrapped method.

        Returns:
            new_kwargs(dict): 'columns' parameter has items from 'columns',
                'prefixes', and 'mask' parameters combined into a single list
                of column names using the 'create_column_list' method.
        """
        # kwargs names to use to create finalized 'columns' argument
        new_kwargs = {}
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            argspec = getfullargspec(method)
            unpassed_args = argspec.args[len(args):]
            if ('columns' in unpassed_args
                    and 'prefixes' in unpassed_args
                    and 'mask' in unpassed_args):
                columns = list(self.datatypes.keys())
            else:
                for argument in arguments_to_check:
                    if argument in kwargs:
                        new_kwargs[argument] = kwargs[argument]
                    else:
                        new_kwargs[argument] = None
                    if argument in ['prefixes', 'mask'] and argument in kwargs:
                        del kwargs[argument]
                columns = self.create_column_list(**new_kwargs)
                kwargs.update({'columns' : columns})
            return method(self, **kwargs)
        return wrapper

    """ Private Methods """

    def _check_wildcards(self, techniques):
        """Converts 'all' or 'default' techniques to list of techniques.

        Args:
            techniques(list or str): techniques names.

        Returns:
            If 'all', all keys listed in 'options' dict are returned.
            If 'default', 'default_techniques' are returned or, if they don't
                exist, all keys listed in 'options' dict are returned.
            Otherwise, 'techniques' is returned intact.
        """
        if techniques in ['all', ['all']]:
            return self.options.keys()
        elif techniques in ['default', ['default']]:
            if hasattr(self, 'default_techniques') and self.default_techniques:
                return self.default_techniques
            else:
                return self.options.keys()
        else:
            return techniques

    def _finalize_parameters(self):
        """Applies Idea settings or default parameters if none exist.

        If 'parameters' is None, the accessible Idea instance is checked for a
        section matching the 'name' attribute of the class instance. If no
        Idea parameters exist, 'default_parameters are used. If there are no
        'default_parameters', an empty dict is created for parameters.

        If 'runtime_parameters' exist in the subclass those are added to
        'parameters' as well.
        """
        nested_parameters = {}
        if not hasattr(self, 'parameters') or self.parameters is None:
            self.parameters = {}
            for technique in self.listify(self.techniques):
                new_params = self._get_parameters(technique)
                new_params = self._select_parameters(technique, new_params)
                new_params.update(self._get_runtime_parameters(technique))
                nested_parameters.update({technique : new_params})
        self.parameters = nested_parameters
        return self

    def _get_parameters(self, technique):
        if hasattr(self, 'parameters') and self.parameters:
            return self.parameters[technique]
        elif technique in self.idea.configuration:
            return {technique : self.idea.configuration[technique]}
        elif self.name in self.idea.configuration:
            return {technique : self.idea.configuration[self.name]}
        elif hasattr(self, 'default_parameters') and self.default_parameters:
            return {technique : self.default_parameters[technique]}
        else:
            return {}

    def _get_runtime_parameters(self, technique):
        if hasattr(self, 'runtime_parameters') and self.runtime_parameters:
            return self.runtime_parameters[technique]
        elif (hasattr(self, 'default_runtime_parameters')
                and self.default_runtime_parameters):
            return {technique : self.default_runtime_parameters[technique]}
        else:
            return {}

    def _nestify_parameters(self):
        for parameters in self.check_nests:
            if hasattr(self, parameters):
                setattr(self, parameters, self.nestify(
                        keys = self.listify(self.techniques),
                        dictionaries = getattr(self, parameters)))
        return self

    def _select_parameters(self, technique, parameters,
                           parameters_to_use = None):
        """For subclasses that only need a subset of the parameters stored in
        idea, this function selects that subset.

        Args:
            parameters_to_use(list or str): list or string containing names of
                parameters to include in final parameters dict.
        """
        if hasattr(self, 'selected_parameters') and self.selected_parameters:
            if not parameters_to_use:
                if isinstance(self.selected_parameters, list):
                    parameters_to_use = self.selected_parameters
                elif (hasattr(self, 'default_parameters')
                        and isinstance(self.default_parameters, dict)):
                    parameters_to_use = list(
                            self.default_parameters[technique].keys())
            new_parameters = {}
            for key, value in parameters.items():
                if key in self.listify(parameters_to_use):
                    new_parameters.update({key : value})
        return new_parameters

    """ Core siMpLify Public Methods """

    def draft(self):
        """Default draft method which sets bare minimum requirements.

        This default draft should only be used if users are planning to manually
        add all options and parameters to the SimpleStep subclass.
        """
        if not hasattr(self, 'options'):
            self.options = {}
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        self.checks = ['idea']
        return self

    def edit_parameters(self, parameters):
        """Adds a parameter set to parameters dictionary.

        Args:
            parameters(dict): dictionary of parameters to be added to
                'parameters' of subclass.

        Raises:
            TypeError: if 'parameters' is not dict type.
        """
        if isinstance(parameters, dict):
            if not hasattr(self, 'parameters') or self.parameters is None:
                self.parameters = parameters
            else:
                self.parameters.update(parameters)
            return self
        else:
            error = 'parameters must be a dict type'
            raise TypeError(error)

    def finalize(self):
        """Finalizes parameters and adds 'parameters' to 'algorithm'."""
        self.techniques = self._check_wildcards(techniques = self.techniques)
        self._nestify_parameters()
        self._finalize_parameters()
        if self.technique != 'none':
            self.algorithm = self.options[self.technique](**self.parameters)
        else:
            self.algorithm = None
        return self

    def produce(self, ingredients, plan = None):
        """Generic implementation method for SimpleStep subclass.

        This method should only be used if the algorithm is to be applied to
        'x' and 'y' in ingredients and sklearn compatible 'fit' and 'transform'
        methods are available.

        Args:
            ingredients(Ingredients): an instance of Ingredients or subclass.
            plan(SimplePlan subclass or instance): is not used by the generic
                method but is made available as an optional keyword for
                compatibility with other 'produce'  methods. This parameter is
                used when the current SimpleStep subclass needs to look back at
                previous SimpleSteps (as in Cookbook steps).
        """
        if self.algorithm:
            self.algorithm.fit(ingredients.x, ingredients.y)
            ingredients.x = self.algorithm.transform(ingredients.x)
        return ingredients

    """ Scikit-Learn Compatibility Methods """

    def fit(self, x, y = None):
        """Generic fit method for partial compatibility to sklearn."""
        # Local import to avoid circular dependency.
        from simplify import Ingredients
        if hasattr(self.algorithm, 'fit'):
            if isinstance(x, pd.DataFrame):
                if y is None:
                    self.algorithm.fit(x)
                else:
                    self.algorithm.fit(x, y)
            elif isinstance(x, Ingredients):
                if y is None:
                    self.algorithm.fit(x.x_train)
                else:
                    self.algorithm.fit(x.x_train, y.y_train)
        else:
            pass
        return self

    def fit_transform(self, x, y = None):
        """Generic fit_transform method for partial compatibility to sklearn.
        """
        self.fit(x = x, y = y)
        self.transform(x = x, y = y)
        return x

    def transform(self, x, y = None):
        """Generic transform method for partial compatibility to sklearn."""
        # Local import to avoid circular dependency.
        from simplify import Ingredients
        if hasattr(self.algorithm, 'transform'):
            if isinstance(x, pd.DataFrame):
                if y is None:
                    x = self.algorithm.transform(x)
                else:
                    x = self.algorithm.transform(x, y)
            elif isinstance(x, Ingredients):
                if y is None:
                    x.x_train = self.algorithm.transform(x.x_train)
                else:
                    x.x_train = self.algorithm.transform(x.x_train, y.y_train)
        else:
            pass
        return x