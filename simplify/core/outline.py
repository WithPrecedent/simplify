"""
.. module:: outline
:synopsis: base class for object creation instructions
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List, Optional, Union


@dataclass
class Outline(Container):
    """Base class for object construction instructions.

    Ideally, this class should have no additional methods beyond the lazy
    loader ('load' method).

    Users can use the idiom 'x in Outline' to check if a particular attribute
    exists and is not None. This means default values for optional arguments
    should generally be set to None to allow use of that idiom.

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

    """ Required ABC Methods """

    def __contains__(self, attribute: str) -> bool:
        """Returns whether attribute exists in a subclass instance.

        Args:
            attribute (str): name of attribute to check.

        Returns:
            bool: whether the attribute exists and is not None.

        """
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    """ Public Methods """

    def load(self) -> object:
        """Returns object from module based upon instance attributes.

        Returns:
            object from module indicated in passed Outline instance.

        """
        return getattr(import_module(self.module), self.component)


@dataclass
class BookOutline(Outline):
    """Contains settings for creating a Book and Chapters.

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
    steps: Optional[Union[List[str], str]] = None
    chapter_type: Optional['Chapter'] = Chapter
    chapter_iterable: Optional[str] = 'chapters'
    metadata: Optional[Dict[str, Any]] = None
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'


@dataclass
class PageOutline(Outline):
    """Contains settings for creating an Algorithm and Parameters.

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
    default: Optional[Dict[str, Any]] = None
    required: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, str]] = None
    selected: Optional[Union[bool, List[str]]] = False
    conditional: Optional[bool] = False
    data_dependent: Optional[Dict[str, str]] = None