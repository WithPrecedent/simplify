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

try:
    import pathos.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

import numpy as np
import pandas as pd

import simplify
from simplify import core
from simplify.core import utilities


@dataclasses.dataclass
class Instructions(core.SimpleLoader):
    """Instructions for 'Worker' construction and usage.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.worker'.
        default_module (Optional[str]): a backup name of module where object to
            use is located (can either be a siMpLify or non-siMpLify module).
            Defaults to 'simplify.worker'.
        overview (Optional['Overview']): an instance with an outline of
            strategies selected for a particular 'Worker'. Defaults to an empty
            'Overview' instance.
        book (Optional[str]): name of Book object in 'module' to load. Defaults
            to 'Book'.
        chapter (Optional[str]): name of Chapter object in 'module' to load.
            Defaults to 'Chapter'.
        technique (Optional[str]): name of Book object in 'module' to load.
            Defaults to 'Technique'.
        options (Optional[str]): name of a core.SimpleRepository instance with
            various options available to a particular 'Worker' instance.
            Defaults to an empty core.SimpleRepository.
        data (Optional[str]): name of attribute or key in a 'Project' instance
            'books' to use as a data object to apply methods to. Defaults to
            'dataset'.
        import_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for exporting data objects.
            Defaults to 'processed'.
        comparer (Optional[bool]): whether the 'Worker' has a parallel structure
            allowing for comparison of different alternatives (True) or is a
            singular sequence of steps (False). Defaults to False.

    """
    name: Optional[str] = None
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.worker')
    default_module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.worker')
    overview: Optional['Overview'] = dataclasses.field(
        default_factory = Overview)
    book: Optional[str] = dataclasses.field(
        default_factory = lambda: 'Book')
    chapter: Optional[str] = dataclasses.field(
        default_factory = lambda: 'Chapter')
    technique: Optional[str] = dataclasses.field(
        default_factory = lambda: 'Technique')
    options: Optional[str] = dataclasses.field(
        default_factory = core.SimpleRepository)
    data: Optional[str] = dataclasses.field(
        default_factory = lambda: 'dataset')
    import_folder: Optional[str] = dataclasses.field(
        default_factory = lambda: 'processed')
    export_folder: Optional[str] = dataclasses.field(
        default_factory = lambda: 'processed')
    comparer: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Converts select attributes from strings to objects.
        self._load_attributes()
        return self

    """ Private Methods """

    def _load_attributes(self,
            attributes: Optional[List[str]] = None) -> None:
        """Initializes select attribute classes.

        If 'attributes' is not passed, a default list of attributes is used.

        Args:
            attributes (Optional[List[str]]): attributes to use the lazy loader
                in the 'load' method on. Defaults to None.

        """
        if not attributes:
            attributes = ['book', 'chapter', 'technique', 'options']
        for attribute in attributes:
            setattr(self, attribute, self.load(attribute = attribute))
        return self


@dataclasses.dataclass
class Worker(base.SimpleSystem):
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
    instructions: Optional[Instructions] = dataclasses.field(
        default_factory = Instructions)
    idea: Optional['Idea'] = None
    options: Optional[core.SimpleRepository] = dataclasses.field(
        default_factory = core.SimpleRepository)
    book: Optional[book.Book] = dataclasses.field(
        default_factory = book.Book)
    auto_draft: Optional[bool] = True
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super().__post_init__()
        # Returns appropriate subclass base on 'comparer' attribute of
        # 'instructions'.
        if self.instructions.comparer:
            return Comparer(
                name = self.name,
                instructions = self.instructions,
                idea = self.idea,
                options = self.options,
                book = self.book,
                auto_draft = self.auto_draft,
                auto_publish = self.auto_publish,
                auto_apply = self.auto_apply)
        else:
            return Sequencer(
                name = self.name,
                instructions = self.instructions,
                idea = self.idea,
                options = self.options,
                book = self.book,
                auto_draft = self.auto_draft,
                auto_publish = self.auto_publish,
                auto_apply = self.auto_apply)

    """ Overview Property """

    @property
    def overview(self) -> Dict[str, List[str]]:
        """Returns snapshot of current state of selected options.

        Returns:
            Dict[str, List[str]]: keys are steps and values are lists of
                selected options.

        """
        try:
            return self._overview
        except AttributeError:
            self._overview = self._get_overview()
            return self._overview

    @overview.setter
    def overview(self, overview: Dict[str, List[str]]) -> None:
        """Sets snapshot of selected options.

        Setting 'overview' will affect other methods which use 'overview' to
        identify which options have been selected.

        Args:
            overview (Dict[str, List[str]]): keys are steps and values are lists
                of selected options.

        """
        if (isinstance(overview, dict)
                and all(isinstance(v, list) for v in overview.values())):
            self._overview = overview
        else:
            raise TypeError(f'overview must be dict of lists')
        return self

    @overview.deleter
    def overview(self) -> None:
        """Sets snapshot of selected options to an empty dictionary.

        There are few, if any reasons, to use the 'overview' deleter. It is
        included in case a user wants the option to clear out current selections
        and add more manually.

        """
        self._overview = {}
        return self

    """ Public Methods """

    def publish(self, book: core.SimplePlan) -> core.SimplePlan:
        """Finalizes each technique in each chapter of 'book'.

        Args:
            book ('SimplePlan'): iterable storing different chapter instances.

        Returns:
            SimplePlan: with techniques finalized.

        """
        return [self._publish_techniques(chapter = chapter) for chapter in book]

    """ Private Methods """

    def _get_overview(self) -> Dict[str, List[str]]:
        """Creates dictionary with techniques for each step.

        Returns:
            Dict[str, Dict[str, List[str]]]: dictionary with keys of steps and
                values of lists of techniques.

        """
        steps = self.idea.get_steps(section = self.instructions.name)
        outline = {}
        for step in steps:
            outline[step] = self.idea.get_techniques(
                section = self.instructions.name,
                step = step)
        return outline

    def _publish_techniques(self, chapter: core.SimplePlan) -> core.SimplePlan:
        """

        """
        new_chapter = []
        for step in chapter:
            new_chapter.append(self.expert.publish(step = step))
        return new_chapter


@dataclasses.dataclass
class Comparer(Worker):
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
    instructions: Optional[Instructions] = dataclasses.field(
        default_factory = Instructions)
    idea: Optional['Idea'] = None
    options: Optional[core.SimpleRepository] = dataclasses.field(
        default_factory = core.SimpleRepository)
    book: Optional[book.Book] = dataclasses.field(
        default_factory = book.Book)
    auto_draft: Optional[bool] = True
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super(core.SimpleSystem).__post_init__()
        return self

    """ Public Methods """

    def draft(self) -> core.SimplePlan:
        """Drafts a book with a parallel chapter structure.

        Returns:
            'SimplePlan': configured to spefications in 'instructions'.

        """
        # Creates a 'SimplePlan' instance to store other 'SimplePlan' instances.
        book = core.SimplePlan(name = self.instructions.book)
        # Creates list of steps from 'outline'.
        steps = list(self.overview.keys())
        # Creates 'possible' list of lists of 'techniques'.
        possible = list(self.overview.values())
        # Creates a list of lists of the Cartesian product of 'possible'.
        combinations = list(map(list, itertools.product(*possible)))
        # Creates a 'chapter' for each combination of techniques and adds that
        # 'chapter' to 'book'.
        for i, techniques in enumerate(combinations):
            chapter = core.SimplePlan(name = f'{self.instructions.chapter}_{i}')
            step_techniques = tuple(zip(steps, techniques))
            for technique in step_techniques:
                technique = self.instructions.technique.load()(
                    name = technique[0],
                    technique = technique[1])
                chapter.add(contents = technique)
            book.add(contents = chapter)
        return book



@dataclasses.dataclass
class Sequencer(Worker):
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
    instructions: Optional[Instructions] = dataclasses.field(
        default_factory = Instructions)
    idea: Optional['Idea'] = None
    options: Optional[core.SimpleRepository] = dataclasses.field(
        default_factory = core.SimpleRepository)
    book: Optional[book.Book] = dataclasses.field(
        default_factory = book.Book)
    auto_draft: Optional[bool] = True
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super(core.SimpleSystem).__post_init__()
        return self

    """ Public Methods """

    def draft(self) -> None:
        """Drafts 'Book' instance with a serial 'techniques' structure.

        Args:
            project ('Project'): an instance for a 'Book' instance to be
                modified.

        Returns:
            'Project': with 'Book' instance modified.

        """
        outline = self._get_outline()
        new_steps = []
        for step, techniques in project.overview[self.worker.name].items():
            for technique in techniques:
                new_steps.append((step, technique))
        project[self.worker.name].steps.extend(new_steps)
        return project

    def publish(self, project: 'Project') -> 'Project':
        """Finalizes 'Book' instance in 'project'.

        Args:
            project ('Project'): an instance for a 'Book' instance to be
                modified.

        Returns:
            'Project': with 'Book' instance modified.

        """
        project[self.worker.name] = self._publish_techniques(
            manuscript = project[self.worker.name])
        return project