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
    loader.
    
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
            __class__.__name__ to make such subclassing easier. 
        module (str): name of 'algorithm' where object to incorporate is located 
            (can either be a siMpLify or non-siMpLify object). 
        algorithm (str): name of object within 'module' to load.
             
    """
    name: str
    module: str
    algorithm: str
    
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
        return getattr(import_module(self.module), self.algorithm) 