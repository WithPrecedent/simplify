"""
.. module:: chapters
:synopsis: base class for storing a group of chapters
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.codex import SimpleCodex
from simplify.core.utilities import listify


@dataclass
class Chapters(MutableMapping):
    """Base class for different Chapter instances to be stored.

    Args:
        options ('Options'): an Options or Options subclass instance.
        chapters (Optional[Union['Chapters', Dict[str, 'Page']]]): a dictionary
            with stored Page instance or a completed Chapters instance. This
            should not be ordinarly passed, but is made available if users
            wish to pass a set of Chapters instance. Defaults to an empty dict.
        book (Optional[object]): related Book or subclass instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If 'name' is not provided,
             __class__.__name__.lower() is used instead.

    """
    options: 'Options'
    chapters: Optional[Union['Chapters', Dict[str, 'Page']]] = field(
        default_factory = dict)
    book: Optional[object] = None
    name: Optional[str] = None

    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        # Sets private 'book' attribute.
        self._book = book
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name') or self.name is not None:
            self.name = self.__class__.__name__.lower()
        # Validates 'chapters' argument.
        self._check_chapters()
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in 'chapters'.

        Args:
            item (str): name of key in 'chapters'.

        """
        try:
            del self.chapters[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> 'Chapter':
        """Returns item in 'chapters'.

        If there are no matches, the method searches for a matching wildcard.

        Args:
            item (str): name of key in 'chapters'.

        Returns:
            'Chapter' instance.

        Raises:
            KeyError: if 'item' is not found in 'chapters' and does not match
                a recognized wildcard.

        """
        try:
            return self.chapters[item]
        except KeyError:
            raise KeyError(' '.join([item, 'is not in', self.name]))

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets 'item' in 'chapters' to 'value'.

        Args:
            item (str): name of key in 'chapters'.
            value (Any): value to be paired with 'item' in 'chapters'.

        """
        self.chapters[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'chapters'."""
        return iter(self.chapters)

    def __len__(self) -> int:
        """Returns length of 'chapters'."""
        return len(self.chapters)

    """ Other Dunder Methods """

    def __add__(self, other: Union['Chapters', Dict[str, 'Chapter']]) -> None:
        """Adds Chapter instances to 'chapters'.

        Args:
            other (Union['Chapters', Dict[str, 'Chapter']]): either another
                'Chapters' instance or an 'chapters' dict.

        Raises:
            TypeError: if 'other' is neither a 'Chapters' instance nor a dict.

        """
        self.add(chapters = other)
        return self

    def __iadd__(self, other: Union['Chapters', Dict[str, 'Chapter']]) -> None:
        """Adds Chapter instances to 'chapters'.

        Args:
            other (Union['Chapters', Dict[str, 'Chapter']]): either another
                'Chapters' instance or an 'chapters' dict.

        Raises:
            TypeError: if 'other' is neither a 'Chapters' instance nor a dict.

        """
        self.add(chapters = other)
        return self

    """ Private Methods """

    def _check_chapters(self):
        """Validates type of passed 'chapters' argument.

        Raises:
            TypeError: if 'chapters' is neither a dictionary nor Chapters
                instance or subclass.

        """
        if (isinstance(self.chapters, Chapters)
                or issubclass(self.chapters, Chapters)):
            self = self.chapters
        elif not isinstance(self.chapters, Dict):
            raise TypeError('chapters must be a dict or Chapters type')
        return self

    """ Public Methods """

    def add(self, chapters: Union['Chapters', Dict[str, 'Chapter']]) -> None:
        """Adds Chapter instances to 'chapters'.

        Args:
            chapters (Union['Chapters', Dict[str, 'Chapter']]): either another
                'Chapters' instance or an 'chapters' dict.

        Raises:
            TypeError: if 'chapters' is neither a 'Chapters' instance nor a
                dict.

        """
        try:
            self.chapters.update(chapters.chapters)
        except AttributeError:
            try:
                self.chapters.update(chapters)
            except AttributeError:
                raise TypeError(' '.join(
                    ['addition requires objects to be dict or Chapters type']))
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        # Applies proxy attribute names, if any are set.
        try:
            self = proxify(instance = self, proxies = self.proxies)
        except AttributeError:
            pass
        return self

    def publish(self) -> None:
        """Finalizes instances in 'chapters'."""
        # Instances and publishes all selected options.
        for key, chapter in self.chapters.items():
            chapter.publish()
        return self

    def apply(self, data: object = None, **kwargs) -> None:
        """Calls 'apply' method for published option matching 'technique'.

        Args:
            data (object): object for option to be applied. Defaults
                to None.
            kwargs: any additional parameters to pass to the option's 'apply'
                method.

        """
        for key, chapter in self.chapters.items():
            chapter.apply(data = data, **kwargs)
        return self

    """ Relational Properties """

    @property
    def chapter(self) -> object:
        """Returns associated Chapter or subclass.

        Returns:
            object stored in 'chapter_type'

        """
        return self.chapter_type

    @chapter.setter
    def chapter(self, chapter: object) -> None:
        """Sets associated Chapter or subclass.

        Args:
            book (object): associated Chapter or subclass.

        """
        self.chapter_type = chapter
        return self

    @chapter.deleter
    def chapter(self) -> NotImplementedError:
        raise NotImplementedError('The chapter property cannot be deleted.')

    @property
    def book(self) -> object:
        """Returns related class instance.

        Returns:
            object stored in '_book'.

        """
        return self._book

    @book.setter
    def book(self, book: object) -> None:
        """Sets related class instance.

        Args:
            book (object): related class instance.

        """
        self._book = book
        return self

    @book.deleter
    def book(self) -> None:
        """Changes '_book' to None."""
        self._book = None
        return self


@dataclass
class Chapter(SimpleCodex):
    """Iterator for a siMpLify process.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        steps (Optional[Dict[str, str]]): ordered list of steps to
            use. Each technique should match a key in 'options'. Defaults to
            None.
        options (Optional['CodexOptions']): options passed from Book instance.
        metadata (Optional[Dict[str, Any]], optional): any metadata about
            the Chapter. In projects, 'number' is automatically a key
            created for 'metadata' to allow for better recordkeeping.
            Defaults to None.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. Defaults to True.
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.
        export_folder (Optional[str]): attribute name of folder in 'inventory' for
            serialization of subclasses to be saved. Defaults to 'chapter'.

    """
    name: Optional[str] = 'chapter'
    steps: Optional[Dict[str, str]] = field(default_factory = dict)
    options: Optional['CodexOptions'] = field(default_factory = dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
    auto_publish: Optional[bool] = True
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'
    parent: Optional[object] = None
    children: Optiona[List[object]] = field(default_factory = list)

    def __post_init__(self) -> None:
        self.proxies = {'parent': 'book', 'children': 'pages', 'child': 'page'}
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def apply(self, data: object, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Subclasses should provide their own 'apply' method, if needed.

        Args:
            data (object): data object for methods to be applied.

        """

        self.ingredients = data
        for step, technique in self.steps.items():
            self.ingredients = self.options[step].apply(
                data = self.ingredients,
                **kwargs)
        return self