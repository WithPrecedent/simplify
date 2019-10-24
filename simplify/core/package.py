"""
.. module:: package
:synopsis: iterable builders and containers
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from itertools import product

from simplify.core.base import SimpleClass


@dataclass
class SimplePackage(SimpleClass):
    """Parent class for building and controlling iterable steps.

    This class adds methods useful to create iterators, iterate over user
    options, and transform data or fit models. SimplePackage subclasses define
    construct iterators and process data with those iterators.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    name: str = 'generic_package'
    steps: object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def __iter__(self):
        try:
            return self.iterable.items()
        except AttributeError:
            return self.steps.items()

    """ Private Methods """

    def _check_order(self, override = False):
        """Creates ordering of class steps."""
        if not self.order or override:
            try:
                self.order = self.listify(self._convert_wildcards(
                    self.idea['_'.join(self.name, 'steps')]))
            except KeyError:
                try:
                    self.order = list(self.steps.keys())
                except TypeError:
                    if isinstance(self.steps, list):
                        self.order = self.steps
                    else:
                        error = 'ordercannot be created for' + self.name
                        raise TypeError(error)
        return self

    def _check_steps(self, override = False):
        """Creates steps dict from order and options."""
        if not self.steps or override or isinstance(self.steps, list):
            new_steps = {}
            for step in self.order:
                try:
                    new_steps[step] = self.options[step]
                except KeyError:
                    new_steps[step] = 'none'
        return self

    def _create_plans(self, override = False):
        """Creates cartesian product of all plans."""
        plans = []
        for step in self.order:
            key = step + '_techniques'
            try:
                plans.append(self.listify(self._convert_wildcards(
                        getattr(self, self.name))))
            except AttributeError:
                plans.append(['none'])
        self.plans = list(map(list, product(*plans)))
        return self

    def _publish_plans(self):
        new_plans = {}
        for i, plan in enumerate(self.plans):
            steps = {}
            for j, technique in enumerate(plan):
                steps.update({self.order[j]: technique})
            new_plans.update(
                    {str(i + 1): self.comparer(number = i + 1, steps = steps)})
        self.plans = new_plans
        # for step in self.order:
        #     setattr(self, step, self.options[step](
        #         technique = self.steps[step]))
        return self

    """ Core siMpLify methods """

    def draft(self):
        """Creates initial settings for class based upon Idea settings."""
        self.checks.append('order')
        super().draft()
        if hasattr(self, 'comparer'):
            self._create_plans()
            try:
                setattr(Plan, 'options', self.options)
                setattr(Plan, 'order', self.order)
            except ValueError:
                error = 'Plan is neither found nor set'
                raise ValueError(error)
        return self

    def edit_plans(self, plans):
        """Adds a comparer or list of plans to the attribute named in
        'comparer_iterable'.

        Args:
            plans(dict(str/int: SimplePlan or list(dict(str/int:
                SimplePlan)): plan(s) to be added to the attribute named in
                'comparer_iterable'.
        """
        if isinstance(plans, dict):
            plans = list(plans.values())
            last_num = list(self.plans.keys())[-1:]
        else:
            last_num = 0
        for i, comparer in enumerate(self.listify(plans)):
            self.plans.update({last_num + i + 1: comparer})
        return self

    def publish(self, *args, **kwargs):
        super().publish()
        for step in self.order:
            getattr(self, step).publish(*args, **kwargs)
        return self

    """ Properties """

    @property
    def all(self):
        return list(self.steps.keys())

    @property
    def defaults(self):
        try:
            return self._defaults
        except AttributeError:
            return list(steps.keys())

    @defaults.setter
    def defaults(self, steps):
        self._defaults = steps


@dataclass
class SimplePlan(SimpleClass):
    """Contains steps to be completed in a siMpLify process.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package.
        number (int): number of plan in a sequence - used for recordkeeping
            purposes.
        steps (dict(str: str)): keys are names of steps and values are names
            of techniques to be applied in those steps.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    name: str = 'generic_plan'
    number: int = 0
    steps: object = None

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
        new_steps = {}
        for step, technique in self.steps.items():
            new_steps.update({step: self.options[step](technique = technique)})
            setattr(self, step, new_steps[step])
        self.steps = new_steps
        return self

    def publish(self, variable, *args, **kwargs):
        if hasattr(self, 'variable_to_store'):
            setattr(self, self.variable_to_store, variable)
        for step, technique in self.steps.items():
            variable = technique.implement(variable, *args, **kwargs)
            if self.exists('return_variables'):
                self._infuse_return_variables(instance = getattr(self, step))
        return self