"""
.. module:: editors
:synopsis: constructs manuscripts
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.base import SimplePublisher
from simplify.core.base import SimpleSequence
from simplify.core.manuscripts import Algorithm
from simplify.core.manuscripts import Book
from simplify.core.manuscripts import Parameters
from simplify.core.utilities import listify


@dataclass
class Author(SimplePublisher):
    """Creates Book instances.

    Args:
        project ('Project'): a related Project instance.

    """
    project: 'Project'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_library(self,
            items: Union[List[str], Dict[str, Any], str]) -> None:
        """Converts selected values in 'mapping' dictionary to Classes.

        Args:
            items (Union[List[str], Dict[str, Any], str]): list of keys,
                dictionary, or a string indicating which 'items' should be
                loaded. If a dictionary is passed, its keys will be used to
                find matches in the 'mapping' dictionary.

        """
        if isinstance(items, dict):
            items = list(items.items())
        for item in listify(items):
            try:
                # Lazily loads all selected Resource instances.
                getattr(self, self.mapping)[item] = getattr(
                    self, self.mapping)[item].load()
            except (KeyError, AttributeError):
                pass
        return self

    def _publish_library(self,
            items: Union[List[str], Dict[str, Any], str]) -> None:
        """Loads, creates, and finalizes instances in the active dictionary.

        Args:
            items (Union[List[str], Dict[str, Any], str]): list of keys,
                dictionary, or a string indicating which 'items' should be
                instanced. If a dictionary is passed, its keys will be used to
                find matches in the 'mapping' dictionary.

        """
        if isinstance(items, dict):
            items = list(items.items())
        for item in listify(items):
            try:
                instance = getattr(self, self.mapping)[item](
                    project = self.project)
                instance.publish()
                instance = getattr(self, self.mapping)[item] = instance
            except (KeyError, AttributeError):
                pass
        return self

    def _publish_steps(self, book: 'Book') -> 'Book':
        """Drafts 'steps' for 'book'.

        Args:
            book ('Book'): Book instance to be modified.

        Returns:
            book ('Book'): Book instance with modifications made.

        """
        # Validates 'steps' or attempts to get 'steps' from 'idea'.
        if not book.steps:
            try:
                book.steps = SimpleSequence(
                    sequence = (
                        book.project.idea[book.name]['_'.join(
                            [book.name, 'steps'])]))
            except KeyError:
                book.steps = SimpleSequence()
        elif not isinstance(book.steps, SimpleSequence):
            book.steps = SimpleSequence(sequence = book.steps)
        return book

    def _publish_chapter_metadata(self, number: int) -> Dict[str, Any]:
        """Finalizes metadata for Chapter instance.

        Args:
            number (int): chapter number; used for recordkeeping.

        Returns:
            Dict[str, Any]: metadata dict.

        """
        metadata = {'number': number + 1}
        try:
            metadata.update(book.metadata)
        except AttributeError:
            pass
        return metadata

    def _publish_chapters(self, book: 'Book') -> 'Book':
        """

        Args:
            book ('Book'): Book instance to be modified.

        Returns:
            book ('Book'): Book instance with modifications made.

        """
        # Creates a list of lists of techniques for each step in 'steps'.
        plans = []
        for step in book.steps:
            try:
                key = '_'.join([step, 'techniques'])
                plans.append(listify(self.project.idea[book.name][key]))
            except KeyError:
                plans.append(['none'])
        # Converts 'plans' to a Cartesian product of all possible 'chapters'.
        plans = list(map(list, product(*plans)))
        for i, steps in enumerate(plans):
            book.chapters.add(
                chapters.chapter_type(
                    project = self.project,
                    chapters = chapters,
                    steps = dict(zip(book.steps, steps)),
                    metadata = self._publish_chapter_metadata(number = i)))
        return book

    """ Core siMpLify Methods """

    def draft(self, outline: 'Resource') -> 'Book':
        """Drafts initial attributes and settings of a Book instance.

        Args:
            outline ('Resource'): instructions for Book creation.

        Returns:
            Book instance.

        """
        book = outline.load()
        return book(project = self.project, name = outline.name)

    def publish(self, book: 'Book') -> 'Book':
        """Drafts initial attributes and settings of 'manuscript'.

        Args:
            manuscript ('Book'): Book instance to be modified.

        Returns:
            manuscript ('Book'): Book instance with modifications made.

        """
        book = self._publish_idea(manuscript = book)
        book = self._publish_options(manuscript = book)
        book = self._publish_chapters(manuscript = manuscript)
        return manuscript


@dataclass
class Contributor(SimplePublisher):
    """Creates Chapters and Chapter instances.

    Args:
        project ('Project'): a related Project instance.
        book ('Book'): a related Book instance.

    """
    project: 'Project'
    book: 'Book'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self, chapters: 'Chapters') -> 'Chapters':
        """Drafts initial attributes and settings of 'manuscript'.

        Args:
            chapters ('Chapters'): empty Chapters instance to be modified.

        Returns:
            Chapters instance with modifications made.

        """
        return self._draft_idea(manuscript = chapters)

    def publish(self, chapters: 'Chapters') -> 'Chapters':
        """Drafts initial attributes and settings of 'manuscript'.

        Args:
            manuscript ('Book'): Book instance to be modified.

        Returns:
            manuscript ('Book'): Book instance with modifications made.

        """
        manuscript = self._publish_pages(manuscript = manuscript)
        return manuscript


@dataclass
class Researcher(SimplePublisher):
    """Creates Page instances.

    Args:
        project ('Project'): a related Project instance.
        chapters ('Chapters'): a related Chapters instance.

    """
    project: 'Project'
    chapters: 'Chapters'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_algorithm(self, page: 'Page', technique: str) -> 'Page':
        """Drafts attributes from 'idea'.

        Args:
            page ('Page'): Page instance to be modified.

        Returns:
            page ('Page'): Page instance with modifications made.

        """
        page.algorithm = Algorithm(page = page, technique = technique)
        return page

    def _draft_parameters(self, page: 'Page', technique: str) -> 'Page':
        """Drafts attributes from 'idea'.

        Args:
            page ('Page'): Page instance to be modified.

        Returns:
            page ('Page'): Page instance with modifications made.

        """
        try:
            page.parameters = Parameters(
                page = page,
                technique = technique,
                parameters = self.project.idea['_'.join(
                    [page.technique, 'parameters'])])
        except (KeyError, AttributeError):
            page.parameters = Parameters(
                page = page,
                technique = technique)
        return page

    """ Core siMpLify Methods """

    def draft(self, manuscript: 'Page', technique: str) -> 'Page':
        """Drafts initial attributes and settings of 'manuscript'.

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        manuscript = self._draft_idea(manuscript = manuscript)
        manuscript = self._draft_options(manuscript = manuscript)
        manuscript = self._draft_algorithm(
            manuscript = manuscript,
            technique = technique)
        manuscript = self._draft_parameters(
            manuscript = manuscript,
            technique = technique)
        return manuscript