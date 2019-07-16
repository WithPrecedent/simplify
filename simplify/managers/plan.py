
from dataclasses import dataclass


@dataclass
class Plan(object):
    """Parent class for defining workflow rules in the siMpLify package.

    Attributes:
        steps: dictionary of steps containing the name of the step and
            corresponding classes. Dictionary keys and values should be placed
            in order that they should be completed.
    """
    steps : object = None

    def __post_init__(self):
        return self

    def _check_attributes(self):
        for step in self.steps:
            if not hasattr(self, step):
                error = step + ' has not been passed to plan class.'
                raise AttributeError(error)
        return self

    def _get_parameters(self, step_name):
        parameters = self.menu[step_name]
        return parameters

    def prepare(self):
        self._check_attributes()
        return self

    def start(self, ingredients):
        self.ingredients = ingredients
        for step in self.steps:
            self.ingredients = getattr(self, step).start(
                    self.ingredients, self)
        return self