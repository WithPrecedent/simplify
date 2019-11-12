"""
.. module:: planner
:synopsis: iterable builder
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module
from itertools import product
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simplify.core.base import SimpleComposite
from simplify.core.plan import SimplePlan
from simplify.core.technique import SimpleTechnique
from simplify.core.utilities import listify


@dataclass
class SimplePlanner(SimpleComposite):
    """Base class for building and controlling iterable techniques.

    This class contains methods useful to create iterators and iterate over
    passed arguments based upon user-selected options. SimplePackage subclasses
    construct iterators and process data with those iterators.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        steps (Optional[Union[List[str], str]]): names of techniques to be
            applied. These names should match keys in the 'options' attribute.
            If using the Idea instance settings, this argument should not be
            passed. Default is None.

    """
    name: Optional[str] = 'simple_planner'
    steps: Optional[Union[List[str], str]] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable 'plans'."""
        try:
            return iter(self.plans.items())
        except AttributeError:
            return iter({})

    """ Private Methods """

    def _draft_steps(self) -> None:
        """Gets 'steps' from injected Idea setting or sets to empty dict."""
        if not self.steps:
            try:
                self.steps = getattr(self, '_'.join([self.name, 'techniques']))
            except AttributeError:
                self.steps = {}
        else:
            self.steps = listify(self.steps)
        return self

    def _draft_techniques(self) -> None:
        """Creates 'techniques' containing technique builder instances."""
        for step in self.steps:
            try:
                technique = getattr(import_module(self.options[step][0]),
                        self.options[step][1])()
                technique.research(distributors = [self.ideas, self.depot])
                technique.draft()
                self.add_techniques(techniques = [technique], names = [step])
            except KeyError:
                error = ' '.join([step,
                                  'does not match an option in', self.name])
                raise KeyError(error)
        return self

    def _draft_plans(self) -> None:
        """Creates cartesian product of all plans for 'techniques' of child
        'techniques'."""
        plans = []
        for technique in self.techniques:
            try:
                plans.append(list(technique.techniques.keys()))
            except AttributeError:
                plans.append(['none'])
        self.plans = list(map(list, product(*plans)))
        return self

    def _publish_plan_metadata(self, number: int) -> Dict[str, Any]:
        metadata = {'number': number + 1}
        try:
            metadata.update(self.metadata)
        except AttributeError:
            pass
        return metadata

    def _publish_techniques(self, ingredients: 'Ingredients') -> None:
        """Finalizes all prepared 'techniques'."""
        for technique in self._techniques.values():
            technique.publish(ingredients = ingredients)
        return self

    def _publish_plans(self) -> None:
        """Converts 'plans' from list of lists to SimplePlan(s)."""
        new_plans = {}
        for i, plan in enumerate(self.plans):
            plan_steps = dict(zip(self.steps, plan))
            plan_steps = self._publish_sequence(steps = plan_steps)
            metadata = self._publish_plan_metadata(number = i)
            try:
                new_plans[str(i + 1)] = self.plan_container(
                    metadata = metadata,
                    steps = plan_steps)
            except AttributeError:
                self.plan_container = SimplePlan
                new_plans[str(i + 1)] = self.plan_container(
                    metadata = metadata,
                    steps = plan_steps)
        self.plans = new_plans
        return self

    def _publish_sequence(self,
            steps: Dict[str, str]) -> Dict[str, SimpleComposite]:
        """Completes 'steps' by converting values to technique instances.

        Args:
            steps (Dict[str, str]): dict of step and technique names to be
                converted to step name and step instances.

        """
        published_steps = {}
        for step, technique in steps.items():
            published_steps[steps] = self.techniques[technique]
        return published_steps

    def _extra_processing(self, plan: SimpleComposite,
            data: SimpleComposite) -> Tuple[SimpleComposite, SimpleComposite]:
        return plan, data

    """ Public Import/Export Methods """

    def load_plan(self, file_path: str) -> None:
        """Imports a single recipe from disc and adds it to the class iterable.

        Args:
            file_path: a path where the file to be loaded is located.
        """
        self.edit_plans(
            plans = self.depot.load(
                file_path = file_path,
                file_format = 'pickle'))
        return self

    """ Core siMpLify methods """

    def draft(self) -> None:
        """Creates initial settings for class based upon Idea settings."""
        self.stage = Stage(idea = self.idea)
        for method in ('steps', 'techniques', 'plans'):
            getattr(self, '_'.join(['_draft', method]))()
        return self

    def edit_plans(self, plans: Union[Dict[Union[str, int], 'SimplePlan'],
                                      List['SimplePlan']]) -> None:
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

    def publish(self, ingredients: 'SimpleDistributor', **kwargs) -> None:
        """Applies class methods to 'ingredients'.

        Args:
            data (SimpleComposite): data object for methods to be applied. This
                can be an Ingredients instance, but other compatiable objects
                work as well.

        """
        self._publish_steps(ingredients = ingredients)
        self._publish_plans()
        for number, plan in self.plans.items():
            if self.verbose:
                print('Testing', plan.name, str(number))
            plan.publish(ingredients = ingredients, **kwargs)
            plan, ingredients = self._extra_processing(
                plan = plan,
                ingredients = ingredients)
        return self


@dataclass
class Stage(object):
    """State machine for siMpLify projects.

    Args:
        idea (Idea): an instance of Idea.
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify project. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    idea: 'Idea'
    name: str = 'stage_machine'

    def __post_init__(self) -> None:
        self.draft()
        return self

    """ Dunder Methods """

    def __repr__(self) -> str:
        """Returns string name of 'state'."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string name of 'state'."""
        return self.state

    """ Private Methods """

    def _set_states(self) -> List[str]:
        """Determines list of possible stages.

        Returns:
            List[str]: states possible based upon user selections.

        """
        states = []
        for stage in listify(self.idea['simplify']['simplify_steps']):
            if stage == 'farmer':
                for step in self.idea['farmer']['farmer_techniques']:
                    states.append(step)
            else:
                states.append(stage)
        return states

    """ State Machine Methods """

    def change(self, new_state: str) -> None:
        """Changes 'state' to 'new_state'.

        Args:
            new_state (str): name of new state matching a string in 'states'.

        Raises:
            TypeError: if new_state is not in 'states'.

        """
        if new_state in self.options:
            self.state = new_state
        else:
            error = new_state + ' is not a recognized stage'
            raise TypeError(error)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Initializes state machine."""
        # Sets list of possible states based upon Idea instance options.
        self.options = self._set_states()
        # Sets initial state.
        self.state = self.options[0]
        return self

    def publish(self) -> None:
        """ Returns current state.

        __str__ and __repr__ can also be used to get the current stage.

        """
        return self.state