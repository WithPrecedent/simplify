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
            values of SimpleStep subclasses.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.
    """

    steps: object = None
    manager_name : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def __call__(self, *args, **kwargs):
        """When called as a function, a SimplePlan class or subclass instance
        will return the 'implement' method.
        """
        return self.implement(*args, **kwargs)

    """ Private Methods """

    def _implement_serial(self, *args, **kwargs):
        for step in self.listify(self.idea_setting):
            result = getattr(self, step).implement(*args, **kwargs)
            getattr(self, self.iterable).update({step: result})
        return self

    def _implement_parallel(self, variable = None):

        return self

    def _publish_parallel(self):
        pass
        return self

    def _publish_serial(self):
        for step_name, step_instance in self.options.items():
            if step_name in getattr(self, self.idea_setting):
                setattr(self, step_name, step_instance())
        return self

    """ Core siMpLify Methods """

    def draft(self):
        self.options = {}
        self.checks = ['iterable']
        self.plan_type = 'serial'
        return self

    def publish(self):
        getattr(self, '_publish_' + self.plan_type)()
        return self

    def implement(self, variable, **kwargs):
        """Iterates through SimpleStep techniques 'implement' methods.

        Args:
            variable(any): variable to be changed by serial SimpleManager
                subclass.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        for step, technique in self.steps.items():
            setattr(self, self.iterable, technique.implement(
                    getattr(self, self.data_variable), **kwargs))
        return self
