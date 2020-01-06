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
from simplify.core.book import Book
from simplify.core.book import Chapter
from simplify.core.book import Page
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

    def _publish_chapter_metadata(self,
            book: 'Book',
            number: int) -> Dict[str, Any]:
        """Finalizes metadata for Chapter instance.

        Args:
            book ('Book'): Book instance to be modified.
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
        """Adds Chapter instances to 'book' 'iterable'.

        Args:
            book ('Book'): Book instance to be modified.

        Returns:
            book ('Book'): Book instance with modifications made.

        """
        # Creates a list of lists of techniques for each step in 'steps'.
        book.techniques = []
        for step in book.steps:
            try:
                plans.append(listify(self.project.idea[book.name]['_'.join(
                    [step, 'techniques'])]))
            except KeyError:
                plans.append(['none'])
        # Converts 'plans' to a Cartesian product of all possible 'techniques'.
        plans = list(map(list, product(*book.techniques)))
        # Creates Chapter instance for every combination of step techniques.
        for i, steps in enumerate(plans):
            techniques = dict(zip(book.steps, steps))
            getattr(book, book.iterable).append(
                Chapter(
                    book = book,
                    name = '_'.join(listify(steps)),
                    techniques = techniques,
                    metadata = self._publish_chapter_metadata(
                        book = book,
                        number = i)))
        return book

    def _publish_contents(self, book: 'Book', outline: 'SimpleBook') -> 'Book':
        """Loads and instances Pages to be used by Chapter instances.

        Args:
            book ('Book'): Book instance to be modified.

        Returns:
            book ('Book'): Book instance with modifications made.

        """
        try:
            options = getattr(import_module(outline.module), 'get_options')(
                idea = project.idea)
        except AttributeError:
            options = getattr(import_module(outline.module), 'options')
        book.contents = SimpleContents(
            project = self.project,
            options = options)
        for i, step in enumerate(book.steps):
            if not step in book.contents:
                book.contents[step] = {}
            for technique in listify(book.techniques[i]):
                if not technique in ['none', None]:
                    page = (options[step][technique].load())
                    book.contents[step][technique] = Page(
                        book = book,
                        name = step,
                        technique = technique,
                        outline = options[step][technique])
                    book.contents[step][technique].algorithm = (
                        book.outline.load())
        return book

    """ Core siMpLify Methods """

    def draft(self, package: 'SimpleBook') -> 'Book':
        """Drafts initial attributes and settings of a Book instance.

        Args:
            package ('SimpleBook'): outline for Book creation.

        Returns:
            Book instance.

        """
        # Loads a Book based upon 'package' attributes.
        book = package.load()
        # Creates an empty instance of a Book.
        book = book(project = self.project, name = package.name)
        # Injects attributes from 'idea' into 'book'.
        book = self._draft_idea(manuscript = book)
        return book

    def publish(self, book: 'Book') -> 'Book':
        """Finalizes and prepares 'book' for application.

        Args:
            book ('Book'): Book instance to be modified.

        Returns:
            book ('Book'): Book instance with modifications made.

        """
        # Sets 'library' to 'default_library' if 'library' not passed.
        book = self._publish_library(manuscript = book)
        # Validates 'steps' in 'book'.
        book = self._publish_steps(manuscript = book)
        # Creates 'chapters' for 'book'.
        book = self._publish_chapters(book = book)
        return book


@dataclass
class Contributor(SimplePublisher):
    """Creates Chapter instances.

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

    """ Private Methods """

    def _publish_idea(self, page: 'Page') -> 'Page':
        """Acquires parameters from Idea instance, if no parameters exist.

        Args:
            page ('Page'): Page instance to be modified.

        Returns:
            page  ('Page'): Page instance with modifications made.

        """
        if not self.parameters:
            try:
                page.parameters.update(self.idea['_'.join(
                    [page.name, 'parameters'])])
            except AttributeError:
                try:
                    page.parameters.update(self.idea['_'.join(
                        [page.technique, 'parameters'])])
                except AttributeError:
                    pass
        return page

    def _publish_selected(self, page: 'Page') -> 'Page':
        """Limits parameters to those appropriate to the 'page' 'technique'.

        If 'page.instructionns.selected' is True, the keys from
        'page.outline.defaults' are used to select the final returned
        parameters.

        If 'page.outline.selected' is a list of parameter keys, then only
        those parameters are selected for the final returned parameters.

        Args:
            page ('Page'): Page instance to be modified.

        Returns:
            page  ('Page'): Page instance with modifications made.

        """
        if self.page.outline.selected:
            if isinstance(self.page.outline.selected, list):
                parameters_to_use = self.page.outline.selected
            else:
                parameters_to_use = list(self.page.outline.default.keys())
            new_parameters = {}
            for key, value in page.parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            page.parameters = new_parameters
        return page

    def _publish_required(self, page: 'Page') -> 'Page':
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            page ('Page'): Page instance to be modified.

        Returns:
            page  ('Page'): Page instance with modifications made.

        """
        try:
            page.parameters.update(self.page.outline.required)
        except TypeError:
            pass
        return page

    def _publish_search(self, page: 'Page') -> 'Page':
        """Separates variables with multiple options to search parameters.

        Args:
            page ('Page'): Page instance to be modified.

        Returns:
            page  ('Page'): Page instance with modifications made.

        """
        page.space = {}
        if page.outline.hyperparameter_search:
            new_parameters = {}
            for parameter, values in page.parameters.items():
                if isinstance(values, list):
                    if any(isinstance(i, float) for i in values):
                        page.space.update(
                            {parameter: uniform(values[0], values[1])})
                    elif any(isinstance(i, int) for i in values):
                        page.space.update(
                            {parameter: randint(values[0], values[1])})
                else:
                    new_parameters.update({parameter: values})
            page.parameters = new_parameters
        return page

    def _publish_runtime(self, page: 'Page') -> 'Page':
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        Args:
            page ('Page'): Page instance to be modified.

        Returns:
            page  ('Page'): Page instance with modifications made.

        """
        try:
            for key, value in self.page.outline.runtime.items():
                try:
                    page.parameters.update({key: getattr(page, value)})
                except AttributeError:
                    raise AttributeError(' '.join(
                        ['no matching runtime parameter', key, 'found']))
        except (AttributeError, TypeError):
            pass
        return page

    def _publish_conditional(self, page: 'Page') -> 'Page':
        """Modifies 'parameters' based upon various conditions.

        A related class should have its own '_publish_conditional' method for
        this method to modify 'published'. That method should have a
        'parameters' and 'name' (str) argument and return the modified
        'parameters'.

        Args:
            page ('Page'): Page instance to be modified.

        Returns:
            page  ('Page'): Page instance with modifications made.

        """
        if 'conditional' in page.outline:
            try:
                page.parameters = page.book._publish_conditional(
                    name = self.page.name,
                    parameters = page.parameters)
            except AttributeError:
                pass
        return page

    """ Core siMpLify Methods """

    def draft(self, book: 'Book') -> 'Book':
        """Sets initial attributes for each Chapter instance iteration."""
        # Declares possible 'parameter_types'.
        self.parameter_types = [
            'idea',
            'selected',
            'required',
            # 'search',
            'runtime',
            'conditional']
        new_chapters = []
        for chapter in book:
            # Injects attributes from 'idea' into a Chapter instance.
            new_chapters.append(self._draft_idea(manuscript = chapter))
        book.chapters = new_chapters
        return book

    def publish(self, book: 'Book') -> 'Book':
        """Finalizes parameters and algorithms for each Chapter in 'book'."""
        # Iterates each chapter in 'book'.
        for chapter in book:
            # Iterates each step and technique in each 'chapter'.
            for step in book.steps:
                # Gets appropriate Page instance from 'book'.
                page = book.contents[step][chapter.steps[step]]
                # Adds parameters to each Page instance.
                for parameter in self.parameter_types:
                    chapter[step] = (getattr(self, '_'.join(
                        ['_publish', parameter]))(page = page))
        return book
