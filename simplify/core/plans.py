"""
.. module:: plans
:synopsis: containers for sets of iterable tasks
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.base import SimpleClass


@dataclass
class SimplePlan(SimpleClass):
    """Contains steps to be completed in a siMpLify process.

    Args:
        name(str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package.
        number(int): number of plan in a sequence - used for recordkeeping
            purposes.
        steps(dict(str: str)): keys are names of steps and values are names
            of techniques to be applied in those steps.
        auto_draft(bool): whether to call the 'publish' method when the class
            is instanced.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    name: str = 'generic_plan'
    number: int = 0
    steps: object = None
    auto_draft: bool = True
    auto_publish: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Public Import/Export Methods """

    def save(self, file_path = None, folder = None, file_name = None):
        self.depot.save(
            variable = self,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = 'pickle')
        return

    """ Core siMpLify Methods """

    def draft(self):
        """Sets default values for a siMpLify plan."""
        super().draft()
        return self

    def publish(self):
        """Creates instanced values in 'steps' and local attributes for those
        instanced steps.
        """
        super().publish()
        new_steps = {}
        for step, technique in self.steps.items():
            new_steps.update({step: self.options[step](technique = technique)})
            setattr(self, step, new_steps[step])
        self.steps = new_steps
        return self

    def implement(self, variable, *args, **kwargs):
        if hasattr(self, 'variable_to_store'):
            setattr(self, self.variable_to_store, variable)
        for step, technique in self.steps.items():
            variable = technique.implement(variable, *args, **kwargs)
            if self.exists('return_variables'):
                self._infuse_return_variables(instance = getattr(self, step))
        return self