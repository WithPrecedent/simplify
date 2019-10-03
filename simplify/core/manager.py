"""
.. module:: manager
:synopsis: builder of siMpLify iterable plans
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from itertools import product

#from tensorflow.test import is_gpu_available

from simplify.core.base import SimpleClass


@dataclass
class SimpleManager(SimpleClass):
    """Parent class for siMpLify planners like Cookbook, Almanac, Analysis,
    and Canvas.

    This class adds methods useful to create iterators, iterate over user
    options, and transform data or fit models.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.
    """

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _check_plan_iterable(self):
        """Creates plan iterable attribute to be filled with concrete plans if
        one does not exist."""
        if not self.exists('plan_iterable'):
            self.plan_iterable = 'plans'
            self.plans = {}
        elif not self.exists(self.plan_iterable):
            setattr(self, self.plan_iterable, {})
        return self

    def _create_steps_lists(self):
        """Creates list of lists of all possible steps in 'options'."""
        self.all_steps = []
        for step in self.options.keys():
            # Stores each step attribute in a list.
            if hasattr(self, step + '_technique'):
                setattr(self, step + '_technique',
                        self.listify(getattr(self, step + '_technique')))
            # Stores a list of 'none' if there is no corresponding local
            # attribute.
            else:
                setattr(self, step, ['none'])
            # Adds step to a list of all steps.
            self.all_steps.append(getattr(self, step + '_technique'))
        return self

    def _publish_plans_parallel(self):
        """Creates plan iterable from list of lists in 'all_steps'."""
        # Creates a list of all possible permutations of step lists.
        all_plans = list(map(list, product(*self.all_steps)))
        for i, plan in enumerate(all_plans):
            published_steps = {}
            for j, (step_name, step_class) in enumerate(self.options.items()):
                published_steps.update(
                        {step_name: step_class(technique = plan[j])})
            getattr(self, self.plan_iterable).update(
                    {i + 1: self.plan_class(steps = published_steps,
                                             number = i + 1)})
        return self

    def _publish_plans_serial(self):
        """Creates plan iterable from list of lists in 'all_steps'."""
        for i, (plan_name, plan_class) in enumerate(self.options.items()):
            for steps in self.all_steps[i]:
                getattr(self, self.plan_iterable).update(
                        {i + 1: plan_class(steps = steps)})
        return self

    def _read_parallel(self, variable = None, **kwargs):
        """Method that implements all of the publishd objects on the
        passed variable.

        The variable is returned after being transformed by called methods.

        Args:
            variable(any): any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        for number, plan in getattr(self, self.plan_iterable).items():
            if self.verbose:
                print('Testing', self.name, str(number))
            variable = plan.read(variable, **kwargs)
        return variable

    def _read_serial(self, variable = None, **kwargs):
        """Method that implements all of the publishd objects on the
        passed variable.

        The variable is returned after being transformed by called methods.

        Args:
            variable(any): any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        for number, plan in getattr(self, self.plan_iterable).items():
            variable = plan.read(variable, **kwargs)
        return variable

    """ Core siMpLify methods """

    def draft(self):
        """ Declares defaults for class."""
        self.options = {}
        self.checks = ['steps', 'depot', 'plan_iterable']
        self.state_attributes = ['depot', 'ingredients']
        return self

    def publish(self):
        """Finalizes iterable dict of plans with instanced plan classes."""
        self._create_steps_lists()
        getattr(self, '_publish_plans_' + self.manager_type)()
        return self

    def read(self, variable = None, **kwargs):
        """Method that implements all of the publishd objects on the
        passed variable.

        The variable is returned after being transformed by called methods.

        Args:
            variable(any): any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        variable = getattr(self, '_read_' + self.manager_type)(
                variable = variable, **kwargs)
        return variable