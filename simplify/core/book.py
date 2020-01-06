"""
.. module:: book
:synopsis: standard iterable classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.base import SimpleManuscript
from simplify.core.base import SimpleOutline


@dataclass
class Book(SimpleManuscript):
    """Standard class for top-level siMpLify package iterable storage.

    Args:
        project (Optional['Project']): related Project or subclass instance.
            Defaults to None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        steps (Optional[List[str], str]): ordered list of steps to execute. Each
            step should match a key in 'contents'. If a string is passed, it is
            converted to a 1-item list. Defaults to an empty list.
        contents (Optional['SimpleContents']): stores SimpleOutlines or
            subclasses in a SimpleContents instance which can be iterated in
            'chapters'. Defaults to an empty dictionary.
        chapters (Optional[List['Chapter']]): a list of Chapter instances that
            include a series of SimpleOutline or Page instances to be applied
            to passed data. Defaults to an empty list.
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'chapters'.
        file_format (Optional[str]): file format to export the Book instance.
            Defaults to 'pickle'.
        export_folder (Optional[str]): the name of the attribute in the Project
            Inventory instance which corresponds to the export folder path to
            use when exporting a Book instance. Defaults to 'book'.
        returns_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    project: Optional['Project'] = None
    name: Optional[str] = None
    steps: List[str] = field(default_factory = list)
    contents: Optional['SimpleContents'] = field(default_factory = dict)
    chapters: Optional[List['Chapter']] = field(default_factory = list)
    iterable: Optional[str] = 'chapters'
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'
    returns_data: Optional[bool] = True


@dataclass
class Chapter(SimpleManuscript):
    """Core iterable for sequences of methods to apply to passed data.

    Args:
        book (Optional['Book']): related Book or subclass instance. Defaults to
            None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        techniques (Optional[Dict[str, str]]): keys are names of 'steps' in
            the related 'book'. Values are particular techniques to pass to
            Page or subclasses when instances are created. Defaults to an empty
            dictionary.
        metadata (Optional[Dict[str, Any]]): information needed by particular
            chapter subclasses or for recordkeeping. By default, SimplePublisher
            instances add 'number' as a key and a corresponding integer.
            Defaults to an empty dictionary.
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'pages'.
        file_format (Optional[str]): file format to export the Book instance.
            Defaults to 'pickle'.
        export_folder (Optional[str]): the name of the attribute in the Project
            Inventory instance which corresponds to the export folder path to
            use when exporting a Book instance. Defaults to 'chapter'.
        returns_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    book: Optional['Book'] = None
    name: Optional[str] = None
    techniques: Optional[Dict[str, str]] = field(default_factory = dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
    iterable: Optional[str] = 'pages'
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'chapter'
    returns_data: Optional[bool] = True


@dataclass
class PageOutline(SimpleOutline):
    """Contains settings for creating a Page instance.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify object).
        component (str): name of python object within 'module' to load.

    """
    name: str
    module: str
    component: str
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    conditional: Optional[bool] = False
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)

    """ Core siMpLify Methods """

    def apply(self, data: object) -> None:
        """Completes parameter dictionary by adding data dependent parameters.

        Args:
            data (object): data object with attributes for data dependent
                parameters to be added.

        Returns:
            parameters with any data dependent parameters added.

        """
        if self.page.outline.data_dependents is not None:
            for key, value in self.data_dependents.items():
                try:
                    page.parameters.update({key, getattr(data, value)})
                except KeyError:
                    print('no matching parameter found for', key, 'in',
                        data.name)
        try:
            chapter.parameters = Parameters(
                chapter = chapter,
                technique = technique,
                parameters = self.project.idea['_'.join(
                    [chapter.technique, 'parameters'])])
        except (KeyError, AttributeError):
            chapter.parameters = Parameters(
                chapter = chapter,
                technique = technique)
        return chapter


@dataclass
class Page(SimpleManuscript):
    """Core iterable for sequences of methods to apply to passed data.

    Args:
        book (Optional['Book']): related Book or subclass instance. Defaults to
            None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        technique (Optional[str]): name of particular technique to be used. It
            should correspond to a key in the related 'book' instance. Defaults
            to None.
        outline (Optional['PageOutline']): instructions for creating a Page
            instance. It must be stored in a Page instance to allow for adding
            of any data dependent parameters.
        file_format (Optional[str]): file format to export the Book instance.
            Defaults to 'pickle'.
        export_folder (Optional[str]): the name of the attribute in the Project
            Inventory instance which corresponds to the export folder path to
            use when exporting a Book instance. Defaults to 'chapter'.
        returns_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    book: Optional['Book'] = None
    name: Optional[str] = None
    technique: Optional[str] = None
    outline: Optional['PageOutline'] = None
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'chapter'
    returns_data: Optional[bool] = True

    """ Dunder Methods """

    def __contains__(self, item: str) -> bool:
        """Returns whether 'attribute' is the 'technique'.

        Args:
            item (str): name of item to check.

        Returns:
            bool: whether the 'item' is equivalent to 'technique'.

        """
        return item == self.technique

    def __iter__(self) -> NotImplementedError:
        """Page instances cannot be iterated."""
        raise NotImplementedError('Page instances cannot be iterated.')

    def __len__(self) -> NotImplementedError:
        """Page instances have no length."""
        raise NotImplementedError('Page instances have no length.')