
from dataclasses import dataclass

from .base import SimpleClass
from .tools import listify


@dataclass
class Step(SimpleClass):
    """Parent class for preprocessing and modeling steps in the siMpLify
    package."""

    def __post_init__(self):
        super().__post_init__()
        return self

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