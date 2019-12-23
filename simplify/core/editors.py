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

from simplify.core.base import SimpleEditor
from simplify.core.base import SimpleOptions
from simplify.core.base import SimpleSequence
from simplify.core.manuscripts import Algorithm
from simplify.core.manuscripts import Book
from simplify.core.manuscripts import Parameters
from simplify.core.utilities import listify


@dataclass
class Author(SimpleEditor):
    """Drafts siMpLify class instances.

    Args:
        project ('Project'): a related director class instance.

    """
    project: 'Project'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_idea(self,
            manuscript: Union[
                'Book', 'Chapter', 'Page']) -> Union['Book', 'Chapter', 'Page']:
        """Drafts attributes from 'idea'.

        Args:
            manuscript (Union['Book', 'Chapter', 'Page']): siMpLify class
                instance to be modified.

        Returns:
            manuscript (Union['Book', 'Chapter', 'Page']): siMpLify class
                instance with modifications made.

        """
        sections = ['general', manuscript.name]
        try:
            sections.extend(listify(manuscript.idea_sections))
        except AttributeError:
            pass
        for section in sections:
            try:
                for key, value in self.project.idea[section].items():
                    if not hasattr(manuscript, key):
                        setattr(manuscript, key, value)
            except KeyError:
                pass
        return manuscript

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
        return manuscript

    def _draft_chapters(self, manuscript: 'Book') -> 'Book':
        """Creates cartesian product of all possible 'techniques'.

        Args:
            manuscript ('Book'): Book instance to be modified.

        Returns:
            manuscript ('Book'): Book instance with modifications made.

        """
        # Initiablizes 'chapters' for 'manuscript'.
        manuscript.chapters = []
        # Creates a list of lists of techniques for each step in 'order'.
        for step in manuscript.order:
            try:
                key = '_'.join([step, 'techniques'])
                manuscript.chapters.append(
                    listify(self.project.idea[manuscript.name][key]))
            except KeyError:
                manuscript.chapters.append(['none'])
        return manuscript

    def _draft_options(self,
            manuscript: Union['Book', 'Page']) -> Union['Book', 'Page']:
        """Drafts 'options' for 'manuscript'.

        Args:
            manuscript (Union['Book', 'Page']): siMpLify class instance to be
                modified.

        Returns:
            manuscript (Union['Book', 'Page']): siMpLify class instance with
                modifications made.

        """
        # Validates 'options' or attempts to get them from 'default_options'.
        if not manuscript.options:
            try:
                manuscript.options = manuscript.default_options
            except AttributeError:
                pass
        return manuscript

    def _draft_algorithm(self, manuscript: 'Page', technique: str) -> 'Page':
        """Drafts attributes from 'idea'.

        Args:
            manuscript ('Page'): Page instance to be modified.

        Returns:
            manuscript ('Page'): Page instance with modifications made.

        """
        manuscript.algorithm = Algorithm(
            page = manuscript,
            technique = technique)
        return manuscript

    def _draft_parameters(self, manuscript: 'Page', technique: str) -> 'Page':
        """Drafts attributes from 'idea'.

        Args:
            manuscript ('Page'): Page instance to be modified.

        Returns:
            manuscript ('Page'): Page instance with modifications made.

        """
        try:
            manuscript.parameters = Parameters(
                page = manuscript,
                technique = technique,
                parameters = self.project.idea['_'.join(
                    [manuscript.technique, 'parameters'])])
        except (KeyError, AttributeError):
            manuscript.parameters = Parameters(
                page = manuscript,
                technique = technique)
        return manuscript

    """ Core siMpLify Methods """

    def draft(self,
            manuscript: Union['Book', 'Chapter', 'Page'],
            technique: Optional[str]) -> Union['Book', 'Chapter', 'Page']:
        """Drafts initial attributes and settings of 'manuscript'.

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        # Calls appropriate methods based upon manuscript type.
        if isinstance(manuscript, Book) or issubclass(manuscript, Book):
            manuscript = self._draft_idea(manuscript = manuscript)
            manuscript = self._draft_options(manuscript = manuscript)
            manuscript = self._draft_order(manuscript = manuscript)
            manuscript = self._draft_chapters(manuscript = manuscript)
        elif isinstance(manuscript, Chapter) or issubclass(manuscript, Chapter):
            manuscript = self._draft_idea(manuscript = manuscript)
            manuscript = self._draft_pages(manuscript = manuscript)
        if isinstance(manuscript, Book) or issubclass(manuscript, Book):
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
class Publisher(SimpleEditor):
    """Publishes siMpLify class instances.

    Args:
        project ('Project'): a related director class instance.

    """
    project: 'Project'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    """ Private Methods """


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

    def _publish_chapters(self,
            manuscript: 'SimpleManuscript') -> 'SimpleManuscript':
        """

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        # Converts 'chapters' to a Cartesian product of all possible 'chapters'.
        new_chapters = list(map(list, product(*manuscript.chapters)))
        manuscript.chapters = []
        for i, steps in enumerate(new_chapters):
            manuscript.chapters.append(
                chapter = manuscript.chapter_type(
                    project = self.project,
                    book = manuscript,
                    order = dict(zip(manuscript.order, steps)),
                    metadata = self._publish_chapter_metadata(number = i)))
        return manuscript

    """ Core siMpLify Methods """

    def publish(self, manuscript: 'SimpleManuscript') -> 'SimpleManuscript':
        """Finalizes objects in 'manuscript'.

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        manuscript.options.publish()
        if issubclass(manuscript, Book):
            manuscript = self._publish_chapters(manuscript = manuscript)
        return manuscript


@dataclass
class Worker(SimpleEditor):
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