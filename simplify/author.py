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
class Author(base.SimpleHandler, core.SimpleProxy):
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
        options (Optional[core.SimpleRepository]):
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
    options: Optional['core.SimpleRepository'] = dataclasses.field(
        default_factory = core.SimpleRepository)

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self.proxify(proxy = 'draft', attribute = 'apply')
        return self

    """ Core siMpLify Methods """


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


