"""
.. module:: book
:synopsis: siMpLify project deliverable
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import dataclasses
import importlib
import types
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify import core


@dataclasses.dataclass
class Technique(core.SimpleLoader):
    """Base method wrapper for applying algorithms to data.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        algorithm (Optional[Union[str, object]]): name of object in 'module' to
            load or the process object which executes the primary method of
            a class instance. Defaults to None.
        parameters (Optional[Dict[str, Any]]): parameters to be attached to
            'algorithm' when 'algorithm' is instanced. Defaults to an empty
            dictionary.

    """
    name: Optional[str] = None
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.core')
    algorithm: Optional[Union[str, object]] = None
    parameters: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory = dict)

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        """Returns string representation of a class instance."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string representation of a class instance."""
        return (
            f'siMpLify {self.name} '
            f'algorithm: {self.algorithm} '
            f'parameters: {str(self.parameters)} ')


@dataclasses.dataclass
class Parameters(core.SimpleRepository):
    """Base class for constructing and storing 'Technique' parameters.

    Args:
        contents (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.

    """
    contents: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory = dict)
    defaults: Optional[List[str]] = dataclasses.field(default_factory = list)