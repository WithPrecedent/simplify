
"""
.. module:: plan
:synopsis: simple iterator
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Iterable, List, Union

from simplify.core.base import SimpleClass
from simplify.core.stage import Stage
from simplify.core.utilities import listify


@dataclass
class SimplePlan(SimpleClass):
    """Iterator for a siMpLify process.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        number (int): number of plan in a sequence - used for recordkeeping
            purposes.
        steps (list(SimpleClass)): any

    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    name: str = 'generic_plan'
    metadata: Dict = None
    steps: List = None

    def __post_init__(self) -> None:
        if self.steps is None:
            self.steps = []
        return self

    """ Dunder Methods """

    def __add__(self, steps: Union['SimpleClass', List['SimpleClass']]) -> None:
        """Adds step(s) at the end of 'steps'.

        Args:
            steps ('SimpleClass' or List['SimpleClass']): the next step(s) to be
                added.

        """
        self.add(steps = steps)
        return self

    def __iadd__(self,
                 steps: Union['SimpleClass', List['SimpleClass']]) -> None:
        """Adds step(s) at the end of 'steps'.

        Args:
            steps ('SimpleClass' or List['SimpleClass']): the next step(s) to be
                added.

        """
        self.add(steps = steps)
        return self

    def __radd__(self,
                 steps: Union['SimpleClass', List['SimpleClass']]) -> None:
        """Adds step(s) at the beginning of 'steps'.

        Args:
            steps ('SimpleClass' or List['SimpleClass']): the step(s) to be
                added at the beginning of 'steps'.

        """
        for step in listify(steps).reverse():
            self.steps.insert(step)
        return self

    """ Import/Export Methods """

    def load(self, file_path = None, folder = None, file_name = None) -> None:
        """Loads 'steps' from disc.

        For any arguments not passed, default values stored in the shared Depot
        instance will be used based upon the current 'stage' of the siMpLify
        project.

        Args:
            file_path (str): a complete file path for the file to be loaded.
            folder (str): a path to the folder where the file should be loaded
                from (not used if file_path is passed).
            file_name (str): contains the name of the file to be loaded without
                the file extension (not used if file_path is passed).

        """
        self.steps = self.depot.load(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = 'pickle')
        return self

    def save(self, file_path = None, folder = None, file_name = None) -> None:
        """Exports 'steps' to disc.

        For any arguments not passed, default values stored in the shared Depot
        instance will be used based upon the current 'stage' of the siMpLify
        project.

        Args:
            file_path (str): a complete file path for the file to be saved.
            folder (str): a path to the folder where the file should be saved
                (not used if file_path is passed).
            file_name (str): contains the name of the file to be saved without
                the file extension (not used if file_path is passed).

        """
        self.depot.save(
            variable = self.steps,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = 'pickle')
        return self

    """ Steps Methods """

    def add(self, steps: Union['SimpleClass', List['SimpleClass']]) -> None:
        """Adds step(s) at the end of 'steps'.

        Args:
            steps ('SimpleClass' or List['SimpleClass']): the next step(s) to be
                added.

        """
        self.steps.extend(listify(step))
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        pass

    def publish(self, variable: 'SimpleClass', **kwargs) -> None:
        """Applies 'steps' to passed 'variable'.

        Args:
            variable ('SimpleClass'): a data container or other SimpleClass
                for steps to be applied to.

        """
        setattr(self, variable.name, variable)
        for step in listify(self.steps):
            try:
                if self.change_state:
                    self.stage = step
            except AttributeError:
                pass
            if step.return_variable:
                setattr(self, variable.name, step.publish(
                    variable = self.variable, **kwargs))
            else:
                setattr(self, step.name, step.publish(
                    variable = self.variable, **kwargs))
        return self

    """ Properties """

    @property
    def stage(self) -> str:
        """Returns the shared stage for the overall siMpLify package.

        Returns:
            str: active state.

        """
        try:
            return self._stage
        except AttributeError:
            self._stage = Stage()
            return self._stage

    @stage.setter
    def stage(self, new_stage: str) -> None:
        """Sets the shared stage for the overall siMpLify package

        Args:
            new_stage (str): active state.

        """
        try:
            self._stage.change(new_stage)
        except AttributeError:
            self._stage = Stage()
            self._stage.change(new_stage)
        return self