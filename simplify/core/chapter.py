"""
.. module:: chapter
:synopsis: composite tree class for storing iterables
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.author import SimpleAuthor
from simplify.core.utilities import listify


@dataclass
class Chapter(SimpleAuthor):
    """Iterator for a siMpLify process.

    Args:
        pages (Dict[str, str]): information needed to create Page classes.
            Keys are step names and values are Algorithm keys.
        metadata (Optional[Dict[str, Any]], optional): any metadata about
            the chapter. In projects, 'number' is automatically a key
            created for 'metadata' to allow for better recordkeeping.
            Defaults to None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.

    """
    pages: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None
    name: Optional[str] = 'chapter'
    file_format: str = 'pickle'
    export_folder: str = 'chapter'

    def __post_init__(self) -> None:
        self.proxies = {'book': 'book', 'chapters': 'pages'}
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable for 'pages'."""
        try:
            return iter(self._pages)
        except AttributeError:
            self._pages= {}
            return iter(self._pages)

    """ Private Methods """

    def _get_page(self,
            key: str,
            technique: str,
            ingredients: 'Ingredients') -> 'Page':
        return self.book.authors[key].publish(
            page = technique,
            data = ingredients)

    """ Core siMpLify Methods """

    def draft(self) -> None:
        return self

    def publish(self, data: Optional[object] = None) -> None:
        """Finalizes 'pages'.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.
                'ingredients' needs to be passed if there are any
                'data_dependent' parameters for the included Page instances
                in 'pages'. Otherwise, it need not be passed. Defaults to None.

        """
        new_pages = {}
        for key, technique in self.pages.items():
            page = self._get_page(
                key = key,
                technique = technique,
                data = ingredients)
            page.chapter = self
            page.publish(data = ingredients)
            new_pages[key] = page
        self.pages = new_pages
        return self

    def apply(self, data: object = None, **kwargs) -> None:
        """Applies 'pages' to 'data'.

        Args:
            data (Optional['Ingredients']): an Ingredients instance for 'pages'
                to be applied.
            **kwargs: any paramters to pass to Page 'apply' methods.

        """
        setattr(self, data.name, data)
        for key, page in self.pages.items():
            try:
                self.book.library.stage = key
            except KeyError:
                pass
            setattr(self, data.name, page.apply(
                data = getattr(self, data.name),
                **kwargs))
        return self