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
        techniques (list or str): names of techniques to be applied. These names
            should match keys in the 'options' attribute.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    name: str = 'generic_package'
    techniques: object = None

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

    def _check_order(self, override = False):
        """Creates ordering of class techniques."""
        if not self._exists('order') or override:
            try:
                self.order = listify(self._convert_wildcards(
                    self.idea['_'.join(self.name, 'techniques')]))
            except KeyError:
                try:
                    self.order = list(self.techniques.keys())
                except TypeError:
                    if isinstance(self.techniques, list):
                        self.order = self.techniques
                    else:
                        error = ' '.join(
                            ['order cannot be created for', self.name])
                        raise TypeError(error)
        return self

    def _draft_composers(self, override = False):
        """Creates 'composers' dict from 'order' and 'options'.

        Args:
            override (bool): whether to override preexisting values.

        """
        if not self._exists('composers'):
            self.composers = {}
        if not self.techniques or override or isinstance(self.techniques, list):
            for step in self.order:
                try:
                    self.composers[step] = self.options[step]()
                except KeyError:
                    error = ' '.join([step, 'does not match any technique in',
                                      self.name])
                    raise KeyError(error)
        return self

    def _draft_plans(self):
        """Creates cartesian product of all plans."""
        plans = []
        for step in self.order:
            key = '_'.join([step, 'techniques'])
            try:
                plans.append(listify(
                    self.composer[step]._convert_wildcards(getattr(self, key))))
            except AttributeError:
                plans.append(['none'])
        self.plans = list(map(list, product(*plans)))
        """Converts 'plans' from list of lists to list of SimplePlan or
        SimpleAlgorithms."""
        new_plans = {}
        for i, plan in enumerate(self.plans):
            algorithms = self._draft_sequence(plan = plan)
            try:
                new_plans.update(
                    {str(i + 1): self.comparer(
                        number = i + 1,
                        steps = algorithms)})
            except AttributeError:
                new_plans.update({str(i + 1): algorithms})
        self.plans = new_plans
        return self

    def _draft_sequence(self, plan: str):
        algorithms = {}
        for j, technique in enumerate(plan):
            algorithm = self.composers[self.order[j]].publish(
                technique = technique)
            algorithms.update({self.order[j]: algorithm})
        return algorithms

    def _extra_processing(self, variable: SimpleClass,
                          simple_object: SimpleClass):
        return simple_object

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
        self.checks.extend(['order'])
        super().draft()
        self._draft_composers()
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

    def publish(self, variable: SimpleClass, **kwargs):
        super().publish()
        for number, simple_object in self.plans.items():
            if self.verbose:
                print('Testing', simple_object.name, str(number))
            simple_object.publish(variable, **kwargs)
            simple_object = self._extra_processing(variable, simple_object)
        return self

    """ Properties """

    @property
    def all(self):
        return list(self.techniques.keys())

    @property
    def defaults(self):
        try:
            return self._defaults
        except AttributeError:
            return list(techniques.keys())

    @defaults.setter
    def defaults(self, techniques):
        self._defaults = techniques



@dataclass
class SimplePlan(SimpleClass):
    """Contains techniques to be completed in a siMpLify process.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        number (int): number of plan in a sequence - used for recordkeeping
            purposes.
        steps (dict(str: str)): keys are names of steps and values are
            algorithms to be applied.

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
        pass

    def publish(self, variable: SimpleClass, *args, **kwargs):
        for step, algorithm in self.steps.items():
            setattr(self, algorithm.publish(
                getattr(self, variable.name), *args, **kwargs))
        return self