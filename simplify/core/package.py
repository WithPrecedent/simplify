"""
.. module:: package
:synopsis: iterable builders and containers
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module
from itertools import product
from typing import Any, List, Dict, Union, Tuple

from simplify.core.base import SimpleClass
from simplify.core.step import SimpleStep
from simplify.core.technique import SimpleTechnique
from simplify.core.utilities import listify


@dataclass
class SimplePackage(SimpleClass):
    """Base class for building and controlling iterable techniques and/or
    other packages.

    This class adds methods useful to create iterators and iterate over passed
    arguments based upon user-selected options. SimplePackage subclasses
    construct iterators and process data with those iterators.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        techniques (List[str] or str): names of techniques to be applied. These 
            names should match keys in the 'options' attribute.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    name: str = 'simple_package'
    techniques: Union[List[str], str] = None

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __iter__(self):
        try:
            return self.plans.items()
        except AttributeError:
            pass

    """ Private Methods """

    def _draft_steps(self) -> None:
        """Creates 'steps' dict from 'techniques' and 'options'."""
        self.steps = {}
        for step in listify(self.techniques):
            try:
                self.steps[step] = getattr(
                    import_module(self.options[step][0]), 
                    self.options[step][1])(techniques = getattr(
                        self, '_'.join(step, 'techniques')))
            except KeyError:
                error = ' '.join([step, 
                                  'does not match an option in', self.name])
                raise KeyError(error)
        return self

    def _draft_plans(self) -> None:
        """Creates cartesian product of all plans."""
        plans = []
        for step, instance in self.steps.items():
            try:
                plans.append(list(instance.techniques.keys()))
            except AttributeError:
                plans.append(['none'])
        self.plans = list(map(list, product(*plans)))
        return self
           
    def _publish_steps(self, data: Union[Ingredients, Tuple]) -> None:
        """Finalizes all prepared SimpleTechniques stored in SimpleSteps."""
        new_steps = {}
        for step, instance in self.steps.items():
            instance.publish(data = data)
            new_steps[step] = instance
        self.steps = new_steps
        return self
    
    def _publish_plans(self) -> None:
        """Converts 'plans' from list of lists to SimplePlan(s)."""
        new_plans = {}
        for i, plan in enumerate(self.plans):
            key_steps = dict(zip(list(self.steps.keys(), plan)))
            steps = self._publish_sequence(steps = key_steps)
            new_plans[str(i + 1)] = self.comparer(number = i + 1, steps = steps)
        self.plans = new_plans
        return self

    def _publish_sequence(self, steps: Dict[str, str]) -> Dict[str, SimpleStep]:
        instanced_steps = {}
        for step, technique in self.steps.items():
            instanced_steps[step] = self.techniques[step][technique]
        return instanced_steps

    def _extra_processing(self, plan: SimpleClass,
            data: SimpleClass) -> Tuple[SimpleClass, SimpleClass]:
        return plan, data

    """ Public Import/Export Methods """

    def load_plan(self, file_path):
        """Imports a single recipe from disc and adds it to the class iterable.

        Args:
            file_path: a path where the file to be loaded is located.
        """
        self.edit_plans(iterables = self.depot.load(file_path = file_path,
                                                    file_format = 'pickle'))
        return self

    """ Core siMpLify methods """

    def draft(self):
        """Creates initial settings for class based upon Idea settings."""
        super().draft()
        self.steps = {}
        self._draft_steps()
        self._draft_plans()
        return self

    def edit_plans(self, plans):
        """Adds a comparer or list of plans to the attribute named in
        'comparer_iterable'.

        Args:
            plans (dict(str/int: SimplePlan or list(dict(str/int:
                SimplePlan)): plan(s) to be added to the attribute named in
                'comparer_iterable'.

        """
        if isinstance(plans, dict):
            plans = list(plans.values())
        try:
            last_num = list(self.plans.keys())[-1:]
        except TypeError:
            last_num = 0
        try:
            for i, comparer in enumerate(listify(plans)):
                self.plans.update({last_num + i + 1: comparer})
        except AttributeError:
            self.plans.update({last_num + i + 1: plans})
        return self

    def publish(self, data: SimpleClass, **kwargs):
        super().publish()
        self._publish_steps(data = data)
        self._publish_plans()
        for number, plan in self.plans.items():
            if self.verbose:
                print('Testing', plan.name, str(number))
            plan.publish(data = data, **kwargs)
            plan, data = self._extra_processing(plan = plan, data = data)
        return self


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