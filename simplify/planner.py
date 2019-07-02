
from dataclasses import dataclass
import pickle
import re
import warnings

from itertools import product
import numpy as np
import pandas as pd

from .inventory import Inventory
from .menu import Menu


@dataclass
class Planner(object):
    """Parent class for Cookbook and Harvest to provide shared methods for
    creating a workflow.

    Users can subclass Planner instead of Cookbook or Harvest if they desire
    to implement a completely different type of workflow.
    """

    menu : object
    inventory : object = None

    def __post_init__(self):
        """Implements basic settings for Planner subclasses."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        self._set_defaults()
        if hasattr(self, 'auto_prepare') and self.auto_prepare:
            self.prepare()
        return self

    def _check_inventory(self):
        """Adds a Inventory instance with default menu if one is not passed
        when a Planner subclass is instanced.
        """
        if not self.inventory:
            self.inventory = Inventory(menu = self.menu)
        return self

    def _check_menu(self):
        """Loads menu from an .ini file if a string is passed to menu instead
        of a menu instance.
        """
        if isinstance(self.menu, str):
            self.menu = Menu(file_path = self.menu)
        return self

    def _create_product(self, return_plans = True):
        plans = product(self.step_lists)
        if return_plans:
            return plans
        else:
            self.all_plans = plans
            return self

    def _localize_menu_settings(self):
        """Local attributes are added from the Menu instance."""
        sections = ['general', 'files']
        if hasattr(self, 'name') and self.name in self.menu:
            sections.append(self.name)
        self.menu.localize(instance = self, sections = sections)
        return self

    def _prepare_step_lists(self):
        """Initializes the step classes for use by the Planner subclass."""
        self.step_lists = []
        for step in self.steps.keys():
            setattr(self, step, self._listify(getattr(self, step)))
            self.step_lists.append(getattr(self, step))
        return self

    def _set_defaults(self):
        self._check_menu()
        self._check_inventory()
        self._localize_menu_settings()
        self.set_order()
        return self

    def add_parameters(self, step, parameters):
        """Adds parameter sets to the parameters dictionary of a prescribed
        step. """
        self.steps[step].add_parameters(parameters = parameters)
        return self

    def add_runtime_parameters(self, step, parameters):
        """Adds runtime_parameter sets to the parameters dictionary of a
        prescribed step."""
        self.steps[step].add_runtime_parameters(parameters = parameters)
        return self

    def add_step(self, name, techniques, step_order = None, **kwargs):
        self.steps.update({name : Step(name = name,
                                       techniques = techniques,
                                       **kwargs)})
        setattr(self, name, list(self.techniques.keys()))
        if step_order:
            self.set_order(order = step_order)
        return self

    def add_techiques(self, step, techniques, algorithms):
        """Adds new technique name and corresponding algorithm to the
        techniques dictionary.
        """
        self.steps[step].add_techniques(techniques = techniques,
                                        algorithms = algorithms)
        return self

    def advance(self, step = None):
        if step:
            if step in self.steps:
                self.step = step
            else:
                error = step + ' is not a recognized step in siMpLify'
                raise KeyError(error)
        else:
            steps_list = list(self.steps.keys())
            current_step = steps_list.index(self.step)
            self.step = steps_list[current_step + 1]
        return self

    def prepare(self):
        return self

    def set_order(self, order = None):
        if order:
            self.order = order
        elif hasattr(self, 'steps') and self.steps:
            self.order = list(self.steps.keys())
        return self

    def start(self):
        return self

@dataclass
class Step(object):
    """Parent class for preprocessing and modeling techniques in the siMpLify
    package."""

    name : str = ''
    technique : str = ''
    techniques : object = None

    def __post_init__(self):
        if hasattr(self, 'auto_prepare') and self.auto_prepare:
            self.prepare()
        return self

    def __contains__(self, technique):
        """Checks whether technique is listed in techniques dictionary."""
        if technique in self.techniques:
            return True
        else:
            return False

    def __delitem__(self, technique):
        """Deletes technique and algorithm if technique is in techniques
        dictionary.
        """
        if technique in self.techniques:
            self.techniques.pop(technique)
        else:
            error = technique + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
        return self

    def __getitem__(self, technique):
        """Gets algorithm if technique is in techniques dictionary."""
        if technique in self.techniques:
            return self.techniques[technique]
        else:
            error = technique + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
            return

    def __setitem__(self, technique, algorithm):
        """Adds technique and algorithm to techniques dictionary."""
        if isinstance(technique, str):
            if isinstance(algorithm, object):
                self.techniques.update({technique : algorithm})
            else:
                error = technique + ' must be an algorithm of object type'
                raise TypeError(error)
        else:
            error = technique + ' must be a string type'
            raise TypeError(error)
        return self

    def _check_kwargs(self, kwargs):
        if not kwargs:
            kwargs = {}
        return kwargs

    def _check_lengths(self, variable1, variable2):
        """Checks lists to ensure they are of equal length."""
        if len(self._listify(variable1) != self._listify(variable1)):
            error = 'Lists must be of equal length'
            raise RuntimeError(error)
            return self
        else:
            return True

    def _check_parameters(self):
        """Checks if parameters exists. If not, defaults are used."""
        if not self.parameters:
            self.parameters = self.defaults
        return self

    def _check_variable(self, variable):
        """Checks if variable exists as attribute in class."""
        if hasattr(self, variable):
            return variable
        else:
            error = self.__class__.__name__ + ' does not contain ' + variable
            raise KeyError(error)

    def _combine_lists(self, *args, **kwargs):
        """Combines lists to create a tuple."""
        return zip(*args, **kwargs)

    def _get_feature_names(self, x, y = None):
        """Gets feature names if previously stored by _store_feature_names."""
        x = pd.DataFrame(x, columns = self.x_cols)
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name = self.y_col)
            return x, y
        else:
            return x

    def _get_indices(self, df, columns):
        """Gets column indices for a list of column names."""
        return [df.columns.get_loc(col) for col in columns]

    def _listify(self, variable):
        """Checks to see if the variable are stored in a list. If not, the
        variable is converted to a list or a list of 'none' is created.
        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]

    def _list_to_string(self, variable):
        """Converts a list to a string with a comma and space separating each
        item. The conversion applies whether variable is a simple list or
        pandas series
        """
        if isinstance(variable, pd.Series):
            out_value = variable.apply(', '.join)
        elif isinstance(variable, list):
            out_value = ', '.join(variable)
        else:
            msg = 'Value must be a list or pandas series containing lists'
            raise TypeError(msg)
            out_value = variable
        return out_value

    def _list_type(self, test_list, data_type):
        """Tests whether any item in a list is of the passed data type."""
        return any(isinstance(i, data_type) for i in test_list)

    def _no_breaks(self, variable, in_column = None):
        """Removes line breaks and replaces them with single spaces. Also,
        removes hyphens at the end of a line and connects the surrounding text.
        Takes either string, pandas series, or pandas dataframe as input and
        returns the same.
        """
        if isinstance(variable, pd.DataFrame):
            variable[in_column].str.replace('[a-z]-\n', '')
            variable[in_column].str.replace('\n', ' ')
        elif isinstance(variable, pd.Series):
            variable.str.replace('[a-z]-\n', '')
            variable.str.replace('\n', ' ')
        else:
            variable = re.sub('[a-z]-\n', '', variable)
            variable = re.sub('\n', ' ', variable)
        return variable

    def _no_double_space(self, variable, in_column = None):
        """Removes double spaces and replaces them with single spaces from a
        string. Takes either string, pandas series, or pandas dataframe as
        input and returns the same.
        """
        if isinstance(variable, pd.DataFrame):
            variable[in_column].str.replace('  +', ' ')
        elif isinstance(variable, pd.Series):
            variable.str.replace('  +', ' ')
        else:
            variable = variable.replace('  +', ' ')
        return variable

    def _remove_excess(self, variable, excess, in_column = None):
        """Removes excess text included when parsing text into sections and
        strips text. Takes either string, pandas series, or pandas dataframe as
        input and returns the same.
        """
        if isinstance(variable, pd.DataFrame):
            variable[in_column].str.replace(excess, '')
            variable[in_column].str.strip()
        elif isinstance(variable, pd.Series):
            variable.str.replace(excess, '')
            variable.str.strip()
        else:
            variable = re.sub(excess, '', variable)
            variable = variable.strip()
        return variable

    def _select_parameters(self, parameters_to_use = []):
        """For subclasses that only need a subset of the parameters stored in
        menu, this function selects that subset.
        """
        new_parameters = {}
        if self.parameters:
            for key, value in self.parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key : value})
            self.parameters = new_parameters
        return self

    def _store_feature_names(self, x, y = None):
        """Stores feature names."""
        self.x_cols = list(x.columns.values)
        if isinstance(y, pd.Series):
            self.y_col = self.label
        return self

    def _word_count(self, variable):
        """Returns word court for a string."""
        return len(variable.split(' ')) - 1

    def add_parameters(self, parameters):
        """Adds a parameter set to parameters dictionary."""
        if isinstance(parameters, dict):
            return self.parameters.update(parameters)
        else:
            error = 'parameters must be a dictionary type'
            raise TypeError(error)
            return self

    def add_runtime_parameters(self, parameters):
        """Adds a parameter set to parameters dictionary."""
        if isinstance(parameters, dict):
            return self.runtime_parameters.update(parameters)
        else:
            error = 'parameters must be a dictionary type'
            raise TypeError(error)
            return self

    def add_techniques(self, techniques, algorithms):
        """Adds new technique name and corresponding algorithm to the
        techniques dictionary.
        """
        self.techniques.update(dict(zip(self._listify(techniques),
                                        self._listify(algorithms))))
        return self

    def fit(self, x, y):
        """Generic fit method for partial compatibility to sklearn."""
        self._initialize()
        return self.algorithm.fit(x, y)

    def fit_transform(self, x, y):
        """Generic fit_transform method for partial compatibility to sklearn.
        """
        self.fit(x, y)
        return self.transform(x)

    def load(self, file_name, import_folder = '', prefix = '', suffix = ''):
        """Loads stored ingredient from disc."""
        import_path = self.inventory.path_join(folder = import_folder,
                                           prefix = prefix,
                                           file_name = file_name,
                                           suffix = suffix,
                                           file_type = 'pickle')
        if self.verbose:
            print('Importing', file_name)
        self.algorithm = pickle.load(open(import_path, 'rb'))
        return self

    def prepare(self, select_parameters = False):
        """Adds parameters to algorithm."""
        self._check_parameters()
        if select_parameters:
            self._select_parameters(
                    parameters_to_use = list(self.defaults.keys()))
        if self.runtime_parameters:
            self.parameters.update(self.runtime_parameters)
        if self.technique != 'none':
            self.algorithm = self.techniques[self.technique]
            if self.parameters:
                self.algorithm = self.algorithm(**self.parameters)
            else:
                self.algorithm = self.algorithm()
        return self

    def save(self, file_name, export_folder = '', prefix = '', suffix = ''):
        """Saves ingredient to disc."""
        if self.verbose:
            print('Exporting', file_name)
        export_path = self.inventory.path_join(folder = export_folder,
                                           prefix = prefix,
                                           file_name = file_name,
                                           suffix = suffix,
                                           file_type = 'pickle')
        pickle.dump(self.algorithm, open(export_path, 'wb'))
        return self

    def start(self, x, y = None):
        """Generic implement method for adding ingredients into recipe and
        applying the appropriate algorithm.
        """
        self.initialize()
        if self.algorithm != 'none':
            self.algorithm.fit(x, y)
            x = self.algorithm.transform(x)
        return x

    def transform(self, x):
        """Generic transform method for partial compatibility to sklearn."""
        return self.algorithm.transform(x)

    def prepare(self):
        kwargs = self.parameters
        self.algorithm = self.techniques[self.technique](**kwargs)
        return self

    def start(self, df, source = None, kwargs = None):
        kwargs = self._check_kwargs(kwargs)
        if self.returned_data in ['data']:
            df = self.algorithm.start(df, **kwargs)
            return df
        elif self.returned_data in ['source']:
            df, source = self.algorithm.start(df, source, **kwargs)
            return df, source