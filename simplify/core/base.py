"""
base: core base classes for siMpLify projects
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    
"""
from __future__ import annotations
import dataclasses
import inspect
import pathlib
from types import ModuleType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import numpy as np
import pandas as pd

import simplify
import sourdough


@dataclasses.dataclass
class SimpleTechnique(sourdough.quirks.Loader, sourdough.components.Technique):
    """Base class for primitive objects in a siMpLify composite object.
    
    The 'contents' and 'parameters' attributes are combined at the last moment
    to allow for runtime alterations.
    
    Args:
        contents (Callable, str): core object used by the 'apply' method or a
            str matching a callable object in the algorithms resource. Defaults 
            to None.
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout siMpLify. For example, if a 
            siMpLify instance needs settings from a Settings instance, 'name' 
            should match the appropriate section name in the Settings instance. 
            When subclassing, it is sometimes a good idea to use the same 'name' 
            attribute as the base class for effective coordination between 
            siMpLify classes. 
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'apply' method is called. Defaults to an empty dict.
                                    
    """
    contents: Union[Callable, str] = None
    name: str = None
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)   
    
    """ Initialization Methods """
    
    def __post_init__(self) -> None:
        """Initializes class instance."""
        # Calls parent initialization methods, if they exist.
        try:
            super().__post_init__()
        except AttributeError:
            pass   
          