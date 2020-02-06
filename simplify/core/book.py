"""
.. module:: book
:synopsis: primary siMpLify iterable classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.repository import Repository
from simplify.core.repository import Plan
from simplify.core.technique import Technique
from simplify.core.utilities import listify


@dataclass
class Book(Repository):
    """Standard class for top-level siMpLify package iterable storage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None. If not passed, __class__.__name__.lower() is used.
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'chapters'.
        techiques (Optional['Repository']): a dictionary of options with
            'Technique' instances stored by step. Defaults to an empty
            'Repository' instance.
        chapters (Optional[List[str]]): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty 'Plan'
            instance.
        alters_data (Optional[bool]): whether the Worker instance's 'apply'
            expects data when the Book instance is iterated. If False, nothing
            is returned. If true, 'data' is returned. Defaults to True.

    """
    name: Optional[str] = None
    iterable: Optional[str] = field(default_factory = lambda: 'chapters')
    # steps: Optional[List[str]] = field(default_factory = list)
    # techniques: Optional['Repository'] = field(default_factory = Repository)
    chapters: Optional[Union[List[str], str]] = field(default_factory = list)
    alters_data: Optional[bool] = True

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns class instance iterable."""
        return iter(getattr(self, self.iterable))

    """ Private Methods """

    # def _add_technique(self,
    #         step: str,
    #         technique: Union['Technique', 'Repository', str]) -> None:
    #     """Adds a single 'technique' to 'techniques'.

    #     Args:
    #         step (str): name of step to which 'technique' belongs.
    #         technique (Union['Technique', 'Repository', str]): a single
    #             Technique instance, a key/value pair of 'Repository' instance,
    #             or a string corresponding to a technique in the 'options' of
    #             'task' in 'project'.

    #     """
    #     if isinstance(technique, Technique):
    #         self.techniques[step][technique.name] = technique
    #     elif isinstance(technique, str):
    #         self.techniques[step][technique] = (
    #             self.project[self.task].options[technique])
    #     elif isinstance(technique, (dict, Repository)):
    #         self.techniques.update(technique)
    #     else:
    #         raise TypeError('technique must be Technique, Repository, or str')
    #     return self

    """ Public Methods """

    def add_chapters(self, chapters: Union['Chapter', List['Chapter']]) -> None:
        """Adds 'chapters' to class iterable.

        Args:
            chapters (Union['Chapter', List['Chapter']]: a 'Chapter' instance or
                list of such instances.

        """
        self.chapters.extend(listify(chapters, default_empty = True))
        return self

    def add_techniques(self,
            step: str,
            techniques: Union[
                'Technique', List['Technique'], str, 'Repository']) -> None:
        """Adds 'techniques' to 'techniques' attribute.

        Args:
            step (str): name of step to which 'techniques' belong.
            techniques (Union['Technique', List['Technique'], str,
                'Repository']): a 'Repository' instance or one or more
                'Technique' instances. If one or more 'Technique' instances are
                passed, the 'name' and 'technique' attributes are used as the
                keys in the 'techniques' attribute.

        """
        if isinstance(techniques, (list, str, Technique)):
            for technique in listify(techniques):
                self._add_technique(step = step, technique = technique)
        elif issubclass(techniques, 'Repository'):
            self.techniques.update(techniques)
        else:
            raise TypeError(
                'technique must be Technique, Repository, list, or str')
        return self

    """ Iterable Proxy Property """

    @property
    def chapters(self) -> List[str]:
        """Returns attribute named in 'iterable'.

        Returns:
            List[str]: with Chapter instances as values.

        """
        return getattr(self, self.iterable)

    @chapters.setter
    def chapters(self, chapters: List[str]) -> None:
        """Sets attribute named in 'iterable' to 'chapters'.

        Args:
            chapters (List[str]): an instance with Chapter instance
                values.

        """
        setattr(self, self.iterable, chapters)
        return self

    @chapters.deleter
    def chapters(self, chapters: Union[List[str], str]) -> None:
        """Deletes 'chapters' from attribute named in 'iterable'.

        Args:
            chapters (Union[List[str], str]): key(s) to Chapter instances to
                be removed from the attribute named in 'iterable'.
        """
        for chapter in listify(chapters):
            try:
                del getattr(self, self.iterable)[chapter]
            except KeyError:
                pass
        return self


@dataclass
class Chapter(Iterable):
    """Standard class for bottom-level siMpLify package iterable storage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'techniques'.
        book (Optional['Book']): related Book or subclass instance. Defaults to
            None.
        number (Optional[int]): number of instance in a sequence. The value is
            used for internal recordkeeping and reporting. Defaults to 0.
        techniques (Optional[Dict[str, str]]): keys are names of 'steps' in
            the related 'book'. Values are particular techniques to pass to
            Technique or subclasses when instances are created. Defaults to an
            empty dictionary.
        alters_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    name: Optional[str] = None
    iterable: Optional[str] = field(default_factory = lambda: 'techniques')
    # book: Optional['Book'] = None
    # number: Optional[int] = 0
    techniques: Optional['Plan'] = field(default_factory = Plan)
    alters_data: Optional[bool] = True

    """ Required ABC Methods """

    def __iter__(self) -> Iterable:
        """Returns class instance iterable."""
        return iter(getattr(self, self.iterable))

    """ Iterable Proxy Property """

    @property
    def techniques(self) -> 'Plan':
        """Returns attribute named in 'iterable'.

        Returns:
            'Plan': with Technique instances as values.

        """
        if self.iterable in ['techniques']:
            return self._techniques
        else:
            return getattr(self, self.iterable)

    @techniques.setter
    def techniques(self, techniques: 'Plan') -> None:
        """Sets attribute named in 'iterable' to 'techniques'.

        Args:
            techniques ('Plan'): an instance with Technique instance
                values.

        """
        if self.iterable in ['techniques']:
            setattr(self, '_techniques', techniques)
        else:
            setattr(self, self.iterable, techniques)
        return self

    @techniques.deleter
    def techniques(self, techniques: Union[List[str], str]) -> None:
        """Deletes 'techniques' from attribute named in 'iterable'.

        Args:
            techniques (Union[List[str], str]): key(s) to Technique instances to
                be removed from the attribute named in 'iterable'.
        """
        for technique in listify(techniques):
            try:
                if self.iterable in ['techniques']:
                    del getattr(self, '_techniques')[technique]
                else:
                    del getattr(self, self.iterable)[technique]
            except KeyError:
                pass
        return self