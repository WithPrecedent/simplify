
from dataclasses import dataclass


@dataclass
class Countertop(object):
    """Parent class for various classes in the siMpLify package to allow
    sharing of methods.
    """

    def _check_lengths(self, variable1, variable2):
        """Checks lists to ensure they are of equal length."""
        if len(self._listify(variable1) != self._listify(variable1)):
            error = 'Lists must be of equal length'
            raise RuntimeError(error)
            return self
        else:
            return True

    def _check_variable(self, variable):
        """Checks if variable exists as attribute in class."""
        if hasattr(self, variable):
            return variable
        else:
            error = self.__class__.__name__ + ' does not contain ' + variable
            raise KeyError(error)

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