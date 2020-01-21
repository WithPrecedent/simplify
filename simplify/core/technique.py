
"""
.. module:: technique
:synopsis: siMpLify algorithms and parameters
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.types import Outline


@dataclass
class TechniqueOutline(Outline):
    """Contains settings for creating a Technique instance.

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
        algorithm: str = None
        default: Optional[Dict[str, Any]] = field(default_factory = dict)
        required: Optional[Dict[str, Any]] = field(default_factory = dict)
        runtime: Optional[Dict[str, str]] = field(default_factory = dict)
        selected: Optional[Union[bool, List[str]]] = False
        conditional: Optional[bool] = False
        data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)

    """
    name: str
    module: str
    algorithm: str = None
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)
    fit_method: Optional[str] = None
    train_method: Optional[str] = None


@dataclass
class Technique(Container):
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
        returns_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    name: Optional[str] = None
    technique: Optional[str] = None
    algorithm: Optional[object] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)
    parameter_space: Optional[Dict[str, List[Union[int, float]]]] = field(
        default_factory = dict)
    data_dependents: Optional[Dict[str, str]] = field(default_factory = dict)
    returns_data: Optional[bool] = True

    """ Dunder Methods """

    def __contains__(self, key: str) -> bool:
        """Returns whether 'attribute' is the 'technique'.

        Args:
            key (str): name of item to check.

        Returns:
            bool: whether the 'key' is equivalent to 'technique'.

        """
        return item == self.technique