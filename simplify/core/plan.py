"""
.. module:: plan
:synopsis: container for siMpLify iterable
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.base import SimpleClass


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
        will return the 'read' method.
        """
        return self.read(*args, **kwargs)

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

    def read(self, variable, **kwargs):
        """Iterates through SimpleStep techniques 'read' methods.

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
            setattr(self, self.data_variable, technique.read(
                    getattr(self, self.data_variable), **kwargs))
        return self
