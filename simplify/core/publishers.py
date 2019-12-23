"""
.. module:: editors
:synopsis: constructs and applies manuscripts
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from itertools import product
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.base import SimplePublisher
from simplify.core.base import SimpleSequence
from simplify.core.manuscripts import Algorithm
from simplify.core.manuscripts import Book
from simplify.core.manuscripts import Parameters
from simplify.core.utilities import listify


@dataclass
class Editor(SimplePublisher):
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

    def _draft_order(self, manuscript: 'Book') -> 'Book':
        """Drafts 'order' for 'manuscript'.

        Args:
            manuscript ('Book'): Book instance to be modified.

        Returns:
            manuscript ('Book'): Book instance with modifications made.

        """
        # Validates 'order' or attempts to get 'order' from 'idea'.
        if not manuscript.order:
            try:
                manuscript.order = SimpleSequence(
                    sequence = (
                        manuscript.project.idea[manuscript.name]['_'.join(
                            [manuscript.name, 'steps'])]))
            except KeyError:
                manuscript.order = SimpleSequence()
        elif not isinstance(manuscript.order, SimpleSequence):
            manuscript.order = SimpleSequence(sequence = manuscript.order)
        return manuscript

    def _draft_chapters(self, manuscript: 'Book') -> 'Book':
        """Creates cartesian product of all possible 'techniques'.

        Args:
            manuscript ('Book'): Book instance to be modified.

        Returns:
            manuscript ('Book'): Book instance with modifications made.

        """
        # Initiablizes 'chapters' for 'manuscript'.
        manuscript.chapters = Chapters()
        return manuscript


    def _publish_chapter_metadata(self, number: int) -> Dict[str, Any]:
        """Finalizes metadata for Chapter instance.

        Args:
            number (int): chapter number; used for recordkeeping.

        Returns:
            Dict[str, Any]: metadata dict.

        """
        metadata = {'number': number + 1}
        try:
            metadata.update(manuscript.metadata)
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
        # Creates a list of lists of techniques for each step in 'order'.
        plans = []
        for step in book.order:
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
                    order = dict(zip(book.order, steps)),
                    metadata = self._publish_chapter_metadata(number = i)))
        return book

    """ Core siMpLify Methods """

    def draft(self, manuscript: 'Book') -> 'Book':
        """Drafts initial attributes and settings of 'manuscript'.

        Args:
            manuscript ('Book'): Book instance to be modified.

        Returns:
            manuscript ('Book'): Book instance with modifications made.

        """
        manuscript = self._draft_idea(manuscript = manuscript)
        manuscript = self._draft_options(manuscript = manuscript)
        manuscript = self._draft_order(manuscript = manuscript)
        return manuscript

    def publish(self, manuscript: 'Book') -> 'Book':
        """Drafts initial attributes and settings of 'manuscript'.

        Args:
            manuscript ('Book'): Book instance to be modified.

        Returns:
            manuscript ('Book'): Book instance with modifications made.

        """
        manuscript = self._publish_options(manuscript = manuscript)
        manuscript = self._publish_chapters(manuscript = manuscript)
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

@dataclass
class Worker(object):
    """Applies methods to siMpLify class instances.

    Args:
        project ('Project'): a related director class instance.

    """
    project: 'Project'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    def _apply_gpu(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients', 'SimpleManuscript']] = None,
            **kwargs) -> NotImplementedError:
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's 'apply' method.

        Raises:
            NotImplementedError: until dynamic GPU support is added.

        """
        raise NotImplementedError(
            'GPU support outside of modeling is not yet supported')

    def _apply_multi_core(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients',
                'SimpleManuscript']] = None) -> 'SimpleManuscript':
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        with Pool() as pool:
            pool.map(manuscript.apply, data)
        pool.close()
        return self

    def _apply_single_core(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients', 'SimpleManuscript']] = None,
            **kwargs) -> 'SimpleManuscript':
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's 'apply' method.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        manuscript.apply(data = data, **kwargs)
        return self

    """ Core siMpLify Methods """

    def apply(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients', 'SimpleManuscript']] = None,
            **kwargs) -> 'SimpleManuscript':
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's options' 'apply' method.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        if self.parallelize and not kwargs:
            self._apply_multi_core(
                manuscript = manuscript,
                data = data)
        else:
            self._apply_single_core(
                manuscript = manuscript,
                data = data,
                **kwargs)
        return manuscript