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
        number(int): number of plan in a sequence - used for recordkeeping
            purposes.
        steps(dict(str: str)): keys are names of steps and values are names
            of techniques to be applied in those steps.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """
    number: int = 0
    steps: object = None
    name: str = 'generic_plan'
    auto_publish: bool = True

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
        for step, technique in self.steps.items():
            variable = technique.implement(variable, *args, **kwargs)
            if self.exists('return_variables'):
                self._infuse_return_variables(instance = getattr(self, step))
        return self

@dataclass
class SimpleBatch(SimplePlan):
    """Contains steps to be completed in a siMpLify process.

    Args:
        number(int): number of plan in a sequence - used for recordkeeping
            purposes.
        steps(dict(str: str)): keys are names of steps and values are names
            of techniques to be applied in those steps.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """
    number: int = 0
    steps: object = None
    name: str = 'generic_batch'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self


@dataclass
class SimpleSequence(SimplePlan):
    """Contains steps to be completed in a siMpLify process.

    Args:
        number(int): number of plan in a sequence - used for recordkeeping
            purposes.
        steps(dict(str: str)): keys are names of steps and values are names
            of techniques to be applied in those steps.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """
    number: int = 0
    steps: object = None
    name: str = 'generic_sequence'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self