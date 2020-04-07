"""
.. module:: Worker
:synopsis: generic siMpLify manager
:publisher: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections.abc
import dataclasses
import importlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core import base
from simplify.core import utilities


@dataclasses.dataclass
class Author(base.SimpleHandler, base.SimpleProxy):
    """Generic subpackage controller class for siMpLify data projects.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        idea (Optional['Idea']): shared project configuration settings.
        options (Optional['SimpleRepository']):
        book (Optional['Book']):
        auto_draft (Optional[bool]): whether to call the 'draft' method when
            instanced. Defaults to True.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            instanced. Defaults to True.
        auto_apply (Optional[bool]): whether to call the 'apply' method when
            instanced. For auto_apply to have an effect, 'dataset' must also
            be passed. Defaults to False.

    """
    name: Optional[str] = None
    idea: Optional['configuration.Idea'] = None
    options: Optional['base.SimpleRepository'] = dataclasses.field(
        default_factory = base.SimpleRepository)

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self.proxify(proxy = 'draft', attribute = 'apply')
        return self

    """ Core siMpLify Methods """

    def outline(self) -> Dict[str, List[str]]:
        """Creates dictionary with techniques for each step.

        Returns:
            Dict[str, Dict[str, List[str]]]: dictionary with keys of steps and
                values of lists of techniques.

        """
        steps = self._get_settings(
            section = self.name,
            prefix = self.name,
            suffix = 'steps')
        return {
            step: self._get_settings(
                section = self.name,
                prefix = step,
                suffix = 'techniques')
            for step in steps}

    def draft(self) -> None:
        """Activates and instances select attributes."""
        self.book = self.instructions.book()
        # Drafts outline of methods to use.
        if self.instructions.comparer:
            self._draft_comparer()
        else:
            self._draft_sequencer()
        return self

    """ Private Methods """

    def _get_settings(self,
            section: str,
            prefix: str,
            suffix: str) -> List[str]:
        """Returns settings from 'idea' based on 'name' and 'suffix'.

        Args:
            section (str): outer key name in 'idea' section.
            prefix (str); prefix for an inner key name.
            suffix (str): suffix to inner key name in 'idea'.

        Returns:
            List[str]: names of matching workers, steps, or techniques.

        """
        return utilities.listify(self.idea[section]['_'.join([prefix, suffix])])

    def _draft_comparer(self) -> None:
        """Drafts 'Book' instance with a parallel chapter structure.

        Args:
            project ('Project'): an instance for a 'Book' instance to be
                modified.

        Returns:
            'Project': with 'Book' instance modified.

        """
        # Creates list of steps from 'project'.
        steps = list(self.overview[self.name].keys())
        # Creates 'possible' list of lists of 'techniques'.
        possible = list(self.overview[self.name].values())
        # Creates a list of lists of the Cartesian product of 'possible'.
        combinations = list(map(list, itertools.product(*possible)))
        # Creates Chapter instance for every combination of techniques.
        for techniques in combinations:
            step_pairs = tuple(zip(steps, techniques))
            chapter = self.instructions.chapter(steps = step_pairs)
            self.book.chapters.append(chapter)
        return project

    def _draft_sequencer(self) -> None:
        """Drafts 'Book' instance with a serial 'techniques' structure.

        Args:
            project ('Project'): an instance for a 'Book' instance to be
                modified.

        Returns:
            'Project': with 'Book' instance modified.

        """
        new_steps = []
        for step, techniques in project.overview[self.worker.name].items():
            for technique in techniques:
                new_steps.append((step, technique))
        project[self.worker.name].steps.extend(new_steps)
        return project
