
from dataclasses import dataclass
import pickle

from .tools import listify


@dataclass
class Step(object):
    """Parent class for preprocessing and modeling steps in the siMpLify
    package."""

    def __post_init__(self):
        if hasattr(self, '_set_defaults'):
            self._set_defaults()
        if hasattr(self, 'auto_prepare') and self.auto_prepare:
            self.prepare()
        return self

    def __call__(self, *args, **kwargs):
        return self.start(*args, **kwargs)

    def __contains__(self, technique):
        """Checks whether technique is listed in techniques dictionary."""
        if technique in self.options:
            return True
        else:
            return False

    def __delitem__(self, technique):
        """Deletes technique and algorithm if technique is in options
        dictionary.
        """
        if technique in self.options:
            self.options.pop(technique)
        else:
            error = technique + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
        return self

    def __getitem__(self, technique):
        """Gets algorithm if technique is in options dictionary."""
        if technique in self.options:
            return self.options[technique]
        else:
            error = technique + ' is not in ' + self.__class__.__name__
            raise KeyError(error)

    def __repr__(self):
        """Returns the name of the current Step."""
        return self.__str__()

    def __setitem__(self, technique, algorithm):
        """Adds technique and algorithm to options dictionary."""
        if isinstance(technique, str):
            if isinstance(algorithm, object):
                self.options.update({technique : algorithm})
            else:
                error = technique + ' must be an algorithm of object type'
                raise TypeError(error)
        else:
            error = technique + ' must be a string type'
            raise TypeError(error)
        return self

    def __str__(self):
        """Returns lowercase name of class."""
        return self.__class__.__name__.lower()

    def _check_lengths(self, variable1, variable2):
        """Checks lists to ensure they are of equal length."""
        if len(listify(variable1) == listify(variable1)):
            return True
        else:
            error = 'Lists must be of equal length'
            raise RuntimeError(error)

    def _check_parameters(self):
        """Checks if parameters exists. If not, defaults are used. If there
        are no defaults, an empty dict is created for parameters.
        """
        if not hasattr(self, 'parameters') or self.parameters is None:
            if hasattr(self, 'menu') and self.name in self.menu.configuration:
                self.parameters = self.menu.configuration[self.name]
            elif hasattr(self, 'default_parameters'):
                self.parameters = self.default_parameters
            else:
                self.parameters = {}
        return self

    def _check_runtime_parameters(self):
        """Checks if class has runtime_parameters and, if so, adds them to
        the parameters attribute.
        """
        if hasattr(self, 'runtime_parameters') and self.runtime_parameters:
            self.parameters.update(self.runtime_parameters)
        return self

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

    def _set_folders(self):
        self.folder_types = ['import', 'export']
        for folder_type in self.folder_types:
            if hasattr(self, folder_type + '_folder'):
                setattr(self, folder_type + '_folder',
                        getattr(self.inventory, folder_type + '_folder'))
        return self

    def add_options(self, techniques, algorithms):
        """Adds new technique name and corresponding algorithm to the
        techniques dictionary.
        """
        if self._check_lengths(techniques, algorithms):
            if getattr(self, 'options') is None:
                self.options = dict(zip(listify(techniques),
                                        listify(algorithms)))
            else:
                self.options.update(dict(zip(listify(techniques),
                                             listify(algorithms))))
            return self

    def add_parameters(self, parameters):
        """Adds a parameter set to parameters dictionary."""
        if isinstance(parameters, dict):
            if not hasattr(self, 'parameters') or self.parameters is None:
                self.parameters = parameters
            else:
                self.parameters.update(parameters)
            return self
        else:
            error = 'parameters must be a dict type'
            raise TypeError(error)

    def add_runtime_parameters(self, parameters):
        """Adds a parameter set to runtime_parameters dictionary."""
        if isinstance(parameters, dict):
            if (not hasattr(self, 'runtime_parameters')
                    or self.runtime_parameters is None):
                self.runtime_parameters = parameters
            else:
                self.runtime_parameters.update(parameters)
            return self
        else:
            error = 'runtime_parameters must be a dict type'
            raise TypeError(error)

    def load(self, file_path):
        """Loads stored step from disc."""
        if self.verbose:
            print('Importing', file_path)
        step = self.inventory.load(file_path, file_format = 'pickle')
        return step

    def save(self):
        """Saves step to disc."""
        if self.verbose:
            print('Exporting', self.name)
        self.inventory.save(variable = self.algorithm,
                            folder = self.inventory.experiment,
                            file_name = self.name,
                            file_format = 'pickle')
        return self