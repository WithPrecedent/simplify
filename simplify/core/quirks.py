"""
quirks: siMpLify specific quirks using the sourdough mixin architecture
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:


"""
from __future__ import annotations
import abc
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import sourdough


@dataclasses.dataclass
class SimpleBase(sourdough.Base, abc.ABC):
    """Abstract base class for automatic registration of subclasses. 
    
    Even though not technically a Quirk subclass (because Quirk inherits from
    Base), it should be used in the same manner. Base is used throughout the
    project subpackage as a Quirk to automatically store base subclasses.
    
    Any non-abstract subclass will automatically store itself in the class 
    attribute 'library' using the snakecase name of the class as the key.
    
    Any direct subclass will automatically store itself in the class attribute 
    'bases' using the snakecase name of the class as the key.
    
    Args:
        bases (ClassVar[sourdough.Library]): related Library instance that will 
            store direct subclasses (those with Base in their '__bases__'
            attribute) and allow runtime construction and instancing of those
            stored subclasses.
    
    Attributes:
        library (ClassVar[sourdough.Library]): related Library instance that 
            will store concrete subclasses and allow runtime construction and 
            instancing of those stored subclasses. 'library' is automatically
            created when a direct Base subclass (Base is in its '__bases__') is 
            instanced.
            
    Namespaces: library, bases, and __init_subclass__.
    
    """
    bases: ClassVar[sourdough.Library] = sourdough.Library()

    """ Initialization Methods """
    
    def __init_subclass__(cls, **kwargs):
        """Adds 'cls' to 'Validator.converters' if it is a concrete class."""
        super().__init_subclass__(**kwargs)
        if not abc.ABC in cls.__bases__:
            key = sourdough.tools.snakify(cls.__name__)
            # Removes 'Simple' from class name so that the key is consistent
            # with the key name for the class being constructed.
            try:
                key = key.replace('simple_', '')
            except ValueError:
                pass
            sourdough.Validator.converters[key] = cls 