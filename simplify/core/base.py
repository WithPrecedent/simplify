"""
Core abstract base class for the siMplify package.

Contents:

    SimpleClass: abstract base class that serves as the parent class (either
        directly or indirectly) for the majority of classes in the siMpLify
        package to provide a consistent structure and sharing of methods.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import warnings

from more_itertools import unique_everseen
from tensorflow.test import is_gpu_available

from simplify.core.depot import Depot
from simplify.core.idea import Idea
from simplify.core.ingredients import Ingredients


@dataclass
class SimpleClass(ABC):
    """Absract base class for major classes in siMpLify package to support
    a common class structure and allow sharing of universal methods.

    To use the class, a subclass must have the following methods:
        draft: a method which sets the default values for the subclass, and 
            usually includes the self.options dictionary. By default, 'draft' 
            is called when __post_init__ is called from a subclass.
        finalize: a method which, after the user has set all options in the
            preferred manner, constructs the objects which can parse, modify,
            process, or analyze data.
            
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
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Creates idea attribute if a string is passed to Idea when subclass was
        # instanced. Injects attributes from idea settings to subclass.
        if self.__class__.__name__ != 'Idea':
            self._check_idea()
        # Calls draft method to set up class instance defaults.
        self.draft()
        # Runs attribute checks from list in self.checks (if it exists).
        self._run_checks()
        # # Adds staticmethods from base module functions as designated in 
        # # 'statics' attribute.
        # self._add_static_methods()
        # Calls finalize method if it exists and auto_finalize is True.
        if hasattr(self, 'auto_finalize') and self.auto_finalize:
            self.finalize()
            # Calls produce method if it exists and auto_produce is True.
            if hasattr(self, 'auto_produce') and self.auto_produce:
                self.produce()
        return self

    """ Magic Methods """

    def __call__(self, idea, *args, **kwargs):
        """When called as a function, a subclass will return the produce method
        after running __post_init__. Any args and kwargs will only be passed
        to the produce method.

        Parameters:
            idea: an instance of Idea or path where an Idea configuration file
                is located must be passed when a subclass is called as a
                function.
        """
        self.idea = idea
        self.auto_finalize = True
        self.auto_produce = False
        self.__post_init__()
        return self.produce(*args, **kwargs)

    def __contains__(self, item):
        """Checks if item is in 'options'; returns boolean.
        
        Parameters:
            item: item to be searched for in 'options' keys.
        """
        return item in self.options

    def __delitem__(self, item):
        """Deletes item if in 'options' or, if an instance attribute, it is 
        assigned a value of None.
        
        Parameters:
            item: item to be deleted from 'options'.
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

        Parameters:
            attr: attribute sought.
        """
        # Intecepts common dict methods and applies them to 'options' dict.
        if attr in ['clear', 'items', 'pop', 'keys', 'update', 'values']:
            return getattr(self.options, attr)
        elif attr in self.__dict__:
            return self.__dict__[attr]
        elif attr.startswith('__') and attr.endswith('__'):
            raise AttributeError
        else:
            return None

    def __getitem__(self, item):
        """Returns item if item is in self.options or is an atttribute.
        
        Parameters:
            item: item matching dict key or attribute name.
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
        
        Parameters:
            item: 'options' key to be set.
            value: corresponding value to be set for 'item' key in 'options'.
        """
        self.options[item] = value
        return self

    """ Private Methods """

    # def _add_static_methods(self):
    #     if hasattr(self, 'statics') and self.statics is not None:
    #         for static in self.statics:
    #             staticmethod(static)
    #     return self
    
    def _check_depot(self):
        """Adds an Depot instance with default idea if one is not passed
        when subclass is instanced.
        """
        if not hasattr(self, 'depot') or self.depot is None:
            self.depot = Depot(idea = self.idea)
        return self
    
    def _check_gpu(self):
        """If gpu status is not set, checks if the local machine has a GPU
        capable of supporting included machine learning algorithms. Because
        the tensorflow 'is_gpu_available' method is very lenient in counting
        what qualifies, it is recommended to set the 'gpu' attribute directly
        or through an Idea instance.
        """
        if hasattr(self, 'gpu'):
            if self.gpu and self.verbose:
                print('Using GPU')
            elif self.verbose:
                print('Using CPU')
        elif is_gpu_available:
            self.gpu = True
            if self.verbose:
                print('Using GPU')
        else:
            self.gpu = False
            if self.verbose:
                print('Using CPU')
        return self
    
    def _check_idea(self):
        """Loads Idea settings from a file if a string is passed instead of an
        Idea instance. Injects sections of 'idea' to a subclass instance using 
        user settings.
        """
        if hasattr(self, 'idea') and isinstance(self.idea, str):
            self.idea = Idea(file_path = self.idea)
        # Adds attributes to class from appropriate sections of the idea.
        sections = ['general']
        if hasattr(self, 'idea_sections') and self.idea_sections:
            if isinstance(self.idea_sections, str):
                sections.append(self.idea_sections)
            else:
                sections.extend(self.idea_sections)
        if (hasattr(self, 'name')
                and self.name in self.idea.configuration
                and not self.name in sections):
            sections.append(self.name)
        print('sections', sections)
        self.idea.inject(instance = self, sections = sections)
        return self

    def _check_ingredients(self, ingredients = None):
        """Checks if ingredients attribute exists. If so, it determines if it
        contains a file folder, file path, or Ingredients instance. Depending
        upon its type, different actions are taken to actually create an
        Ingredients instance. If ingredients is None, then an Ingredients
        instance is created with no pandas DataFrames or Series within it.

        Parameters:
            ingredients: an Ingredients instance, a file path containing a
                DataFrame or Series to add to an Ingredients instance, or
                a folder containing files to be used to compose Ingredients
                DataFrames and/or Series.
        """
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

    """ Public Methods """

    def conform(self, step):
        """Sets 'step' attribute to current step in siMpLify. This is used
        to maintain a universal state in the package for subclasses that are
        state dependent.
        
        Parameters:
            step: string corresponding to current state in siMpLify package.
        """
        if not step:
            step = self.step
        if hasattr(self, 'state_attributes') and self.state_attributes:
            for attribute in self.state_attributes:
                getattr(self, attribute).conform(step = step)
        return self

    @staticmethod
    def deduplicate(iterable):
        """Deduplicates list, pandas DataFrame, or pandas Series.
        
        Parameters:
            iterable: a list, DataFrame, or Series.
        """
        if isinstance(iterable, list):
            return list(unique_everseen(iterable))
    # Needs implementation for pandas
        elif isinstance(iterable, pd.Series):
            return iterable
        elif isinstance(iterable, pd.DataFrame):
            return iterable
    
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
        """Updates options dictionary with passed arguments.
        
        Parameters:
            techniques: a string name or list of names for keys in the 'options'
                dict.
            algorithms: siMpLify compatible objects which can be integrated in
                the package framework. If they are custom algorithms, they
                should be subclassed from SimpleClass or Algorithm to ensure
                compatibility.
            options: a dictionary with keys of techniques and values of 
                algorithms.
        """
        if options:
            self.name_to_type.update(options)
        if techniques and algorithms:
            self.name_to_type.update(dict(zip(techniques, algorithms)))
        return self
    
    @abstractmethod
    def finalize(self, **kwargs):
        """Required method which creates any objects to be applied to data or
        variables. In the case of iterative classes, such as Cookbook, this
        method should construct any plans to be later implemented by the 
        'produce' method. It is roughly equivalent to the scikit-learn fit
        method.
        
        Parameters:
            **kwargs: keyword arguments are not ordinarily included in the 
                finalize method. But nothing precludes them from being added
                to subclasses.
        """
        pass
        return self

    @staticmethod
    def listify(variable):
        """Checks to see if the variable is stored in a list. If not, the
        variable is converted to a list or a list of 'none' is created if the
        variable is empty.
        
        Parameters:
            variable: either a string or list which will, if needed, be 
                transformed into a list to allow iteration.
        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]
        
    def load(self, name = None, file_path = None, folder = None,
             file_name = None, file_format = None):
        """Loads object from file into subclass attribute.

        Parameters:
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

        Parameters:
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