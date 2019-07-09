
from dataclasses import dataclass
import pickle

from ..implements import listify


@dataclass
class Step(object):
    """Parent class for preprocessing and modeling techniques in the siMpLify
    package."""

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

    def _check_parameters(self):
        """Checks if parameters exists. If not, defaults are used."""
        if not self.parameters:
            if hasattr(self, 'default_parameters'):
                self.parameters = self.default_parameters
            else:
                self.parameters = {}
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

    def _list_type(self, test_list, data_type):
        """Tests whether any item in a list is of the passed data type."""
        return any(isinstance(i, data_type) for i in test_list)

    def _select_parameters(self, parameters_to_use = None):
        """For subclasses that only need a subset of the parameters stored in
        menu, this function selects that subset.
        """
        if hasattr(self, 'selected_parameters') and self.selected_parameters:
            if not parameters_to_use:
                parameters_to_use = list(self.default_parameters.keys())
            new_parameters = {}
            if self.parameters:
                for key, value in self.parameters.items():
                    if key in self.default_parameters:
                        new_parameters.update({key : value})
                self.parameters = new_parameters
        return self

    def add_parameters(self, parameters):
        """Adds a parameter set to parameters dictionary."""
        if isinstance(parameters, dict):
            return self.parameters.update(parameters)
        else:
            error = 'parameters must be a dict type'
            raise TypeError(error)
            return self

    def add_techniques(self, techniques, algorithms):
        """Adds new technique name and corresponding algorithm to the
        techniques dictionary.
        """
        self.options.update(dict(zip(listify(techniques),
                                     listify(algorithms))))
        return self


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

    def save(self, file_name, export_folder = '', prefix = '', suffix = ''):
        """Saves step to disc."""
        if self.verbose:
            print('Exporting', file_name)
        export_path = self.inventory.path_join(folder = export_folder,
                                           prefix = prefix,
                                           file_name = file_name,
                                           suffix = suffix,
                                           file_type = 'pickle')
        pickle.dump(self.algorithm, open(export_path, 'wb'))
        return self