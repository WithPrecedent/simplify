"""
.. module:: publishers
:synopsis: constructs manuscripts
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.base import SimpleEditor
from simplify.core.book import Book
from simplify.core.book import Chapter
from simplify.core.book import Technique
from simplify.core.utilities import listify


@dataclass
class Author(SimpleEditor):
    """Creates Book instances.

    Args:
        project ('Project'): a related Project instance.

    """
    project: 'Project'

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""

        return self

    """ Private Methods """

    def _draft_options(self, outline: 'BookOutline') -> None:
        """Loads and instances Contents to be used by Chapter instances.

        Args:
            outline ('BookOutline'): outline for Book creation.

        """
        try:
            self.contents = getattr(
                import_module(book.__module__), 'get_options')(
                    idea = self.project.idea)
        except AttributeError:
            self.contents = getattr(import_module(book.__module__), 'OPTIONS')
        return self

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
        # Creates a list of lists of algorithms for each step in 'steps'.
        algorithms = []
        for step in book.steps:
            try:
                algorithms.append(listify(self.project.idea[book.name]['_'.join(
                        [step, 'algorithms'])]))
            except KeyError:
                algorithms.append(['none'])
        # Converts 'plans' to a Cartesian product of all possible 'algorithms'.
        plans = list(map(list, product(*algorithms)))
        # Creates Chapter instance for every combination of step algorithms.
        for i, steps in enumerate(plans):
            techniques = dict(zip(book.steps, steps))
            if not hasattr(book, book.iterable):
                setattr(book, book.iterable, [])
            getattr(book, book.iterable).append(
                Chapter(
                    book = book,
                    name = '_'.join(listify(steps)),
                    techniques = techniques,
                    metadata = self._publish_chapter_metadata(
                        book = book,
                        number = i)))
        return book

    def _publish_contents(self, book: 'Book') -> 'Book':
        """

        Args:
            book ('Book'): Book instance to be modified.

        Returns:
            book ('Book'): Book instance with modifications made.

        """
        for i, step in enumerate(book.steps):
            if not step in book.contents:
                book.contents[step] = {}
            for technique in listify(book.techniques[i]):
                if not technique in ['none', None]:
                    technique = (options[step][technique].load())
                    book.contents[step][technique] = Technique(
                        book = book,
                        name = step,
                        technique = technique,
                        outline = options[step][technique])
                    book.contents[step][technique].algorithm = (
                        book.outline.load())
        return book

    """ Core siMpLify Methods """

    def draft(self, outline: 'BookOutline') -> 'Book':
        """Drafts initial attributes and settings of a Book instance.

        Args:
            outline ('BookOutline'): outline for Book creation.

        Returns:
            Book instance.

        """
        # Loads a Book class based upon 'outline' attributes.
        book = outline.load()
        # Creates an empty class instance of a Book.
        book = book(outline = outline, name = outline.name)
        # Creates initial 'contents' with TechniqueOutlines.
        book = self._draft_contents(book = book)
        return book

    def publish(self, book: 'Book') -> 'Book':
        """Finalizes and prepares 'book' for application.

        Args:
            book ('Book'): Book instance to be modified.

        Returns:
            book ('Book'): Book instance with modifications made.

        """
        if isinstance(book, Book):
            book = self._publish_contents(book = book)
        # Validates 'steps' in 'book'.
        book = self._publish_steps(manuscript = book)
        # Creates 'chapters' for 'book'.
        book = self._publish_chapters(book = book)
        return book

@dataclass
class Contributor(SimpleEditor):
    """Creates Chapter instances.

    Args:
        project ('Project'): a related Project instance.
        author ('Author'): a related Author instance.

    """
    project: 'Project'
    author: 'Author'

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _publish_idea(self, technique: 'Technique') -> 'Technique':
        """Acquires parameters from Idea instance, if no parameters exist.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            technique  ('Technique'): Technique instance with modifications made.

        """
        if not technique.parameters:
            try:
                technique.parameters.update(self.project.idea.configuration['_'.join(
                    [technique.name, 'parameters'])])
            except KeyError:
                try:
                    technique.parameters.update(
                        self.project.idea.configuration['_'.join(
                            [technique.technique, 'parameters'])])
                except AttributeError:
                    pass
        return technique

    def _publish_selected(self, technique: 'Technique') -> 'Technique':
        """Limits parameters to those appropriate to the 'technique' 'technique'.

        If 'technique.techniquens.selected' is True, the keys from
        'technique.outline.defaults' are used to select the final returned
        parameters.

        If 'technique.outline.selected' is a list of parameter keys, then only
        those parameters are selected for the final returned parameters.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            technique  ('Technique'): Technique instance with modifications made.

        """
        if technique.outline.selected:
            if isinstance(technique.outline.selected, list):
                parameters_to_use = technique.outline.selected
            else:
                parameters_to_use = list(technique.outline.default.keys())
            new_parameters = {}
            for key, value in technique.parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            technique.parameters = new_parameters
        return technique

    def _publish_required(self, technique: 'Technique') -> 'Technique':
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            technique  ('Technique'): Technique instance with modifications made.

        """
        try:
            technique.parameters.update(technique.outline.required)
        except TypeError:
            pass
        return technique

    def _publish_search(self, technique: 'Technique') -> 'Technique':
        """Separates variables with multiple options to search parameters.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            technique  ('Technique'): Technique instance with modifications made.

        """
        technique.space = {}
        if technique.outline.hyperparameter_search:
            new_parameters = {}
            for parameter, values in technique.parameters.items():
                if isinstance(values, list):
                    if any(isinstance(i, float) for i in values):
                        technique.space.update(
                            {parameter: uniform(values[0], values[1])})
                    elif any(isinstance(i, int) for i in values):
                        technique.space.update(
                            {parameter: randint(values[0], values[1])})
                else:
                    new_parameters.update({parameter: values})
            technique.parameters = new_parameters
        return technique

    def _publish_runtime(self, technique: 'Technique') -> 'Technique':
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            technique  ('Technique'): Technique instance with modifications made.

        """
        try:
            for key, value in technique.outline.runtime.items():
                try:
                    technique.parameters.update({key: getattr(technique, value)})
                except AttributeError:
                    raise AttributeError(' '.join(
                        ['no matching runtime parameter', key, 'found']))
        except (AttributeError, TypeError):
            pass
        return technique

    def _publish_conditional(self, book: 'Book', technique: 'Technique') -> 'Technique':
        """Modifies 'parameters' based upon various conditions.

        A related class should have its own '_publish_conditional' method for
        this method to modify 'published'. That method should have a
        'parameters' and 'name' (str) argument and return the modified
        'parameters'.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            technique  ('Technique'): Technique instance with modifications made.

        """
        if 'conditional' in technique.outline:
            try:
                technique.parameters = book._publish_conditional(
                    name = technique.name,
                    parameters = technique.parameters)
            except AttributeError:
                pass
        return technique

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
            self._draft_idea(manuscript = chapter)
        return book

    def publish(self, book: 'Book') -> 'Book':
        """Finalizes parameters and algorithms for each Chapter in 'book'."""
        # Iterates each chapter in 'book'.
        for chapter in book:
            # Iterates each step and technique in each 'chapter'.
            for step in book.steps:
                # Gets appropriate Technique instance from 'book'.
                if not hasattr(chapter, chapter.iterable):
                    setattr(chapter, chapter.iterable, {})
                technique = getattr(chapter, chapter.iterable)[step]
                if technique in ['none']:
                    pass
                else:
                    outline = book.contents[step][technique]
                    technique = outline.load()
                    technique = Technique(
                        book = book,
                        name = step,
                        technique = technique,
                        outline = outline)
                    # Adds parameters to each Technique instance.
                    for parameter in self.parameter_types:
                        if parameter in ['conditional']:
                            getattr(chapter, chapter.iterable)[step] = (
                                self._publish_conditional(
                                    book = book,
                                    technique = technique))
                        else:
                            getattr(chapter, chapter.iterable)[step] = (
                                getattr(self, '_'.join(
                                    ['_publish', parameter]))(technique = technique))
        return book


@dataclass
class Contents(SimpleOptions):
    """Base class for storing options and strategies.

    Args:
        options (Optional[str, Any]): default stored dictionary. Defaults to an
            empty dictionary.
        null_value (Optional[Any]): value to return when 'none' is accessed.
            Defaults to None.

    """
    options: Optional[Dict[str, Any]] = field(default_factory = dict)
    null_value: Optional[Any] = None

    def __post_init__(self) -> None:
        """Sets name of internal 'lexicon' dictionary."""
        self.wildcards = ['default', 'all', 'none']
        super().__post_init__()
        return self

    """ Wildcard Properties """

    @property
    def none(self) -> None:
        """Returns 'null_value'.

        Returns:
            'null_value' attribute or None.

        """
        return self.null_value

    @none.setter
    def none(self, null_value: Any) -> None:
        """Sets 'none' to 'null_value'.

        Args:
            null_value (Any): value to return when 'none' is sought.

        """
        self.null_value = null_value
        return self
