"""
.. module:: technique
:synopsis: siMpLify project algorithm wrapper
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import simplify
from simplify import core


@dataclasses.dataclass
class SimpleStep(core.SimpleComponent):
    """Base class for storing Techniques stored in SimplePlans.

    A SimpleStep is a basic wrapper for a technique that adds a 'name' for the
    'step' that a stored Technique instance is associated with. This makes the
    process of drafting, publishing, and applying Technique instances to data
    a bit cleaner. Also, subclasses of SimpleStep can store additional methods
    and attributes to apply to all possible Technique instances that could be
    used.

    A SimpleStep instance will try to return attributes from 'technique' if the
    attribute is not found in the SimpleStep instance. Similarly, when setting
    or deleting attributes, a SimpleStep instance will set or delete the
    attribute in

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        technique (Optional[str]): Technique object for this step in a siMpLify
            sequence. Defaults to None.

    """
    name: Optional[str] = None
    technique: [SimpleTechnique] = None

    """ Dunder Methods """

    def __getattr__(self, attribute: str) -> Any:
        """Looks for 'attribute' in 'technique'.

        Args:
            attribute (str): name of attribute to return.

        Returns:
            Any: matching attribute.

        Raises:
            AttributeError: if 'attribute' is not found in 'technique'.

        """
        try:
            return getattr(self.technique, attribute)
        except AttributeError:
            raise AttributeError(f'{attribute} not found in {self.name}')

    def __setattr__(self, attribute: str, value: Any) -> None:
        """Adds 'value' to 'technique' with the name 'attribute'.

        Args:
            attribute (str): name of the attribute to add to 'technique'.
            value (Any): value to store at that attribute in 'technique'.

        """
        setattr(self.technique, attribute, value)
        return self

    def __delattr__(self, attribute: str) -> None:
        """Deletes 'attribute' from 'technique'.

        Args:
            attribute (str): name of attribute to delete.

        """
        try:
            delattr(self.technique, attribute)
        except AttributeError:
            pass
        return self

    def __repr__(self) -> str:
        """Returns string representation of a class instance."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string representation of a class instance."""
        return (
            f'step: {self.name} '
            f'technique: {self.technique.name}')


@dataclasses.dataclass
class SimpleTechnique(core.SimpleLoader):
    """Base class for storing and combining algorithms and parameters.

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

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        if self.name not in ['none', None]:
            self.load('algorithm')
        return self

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