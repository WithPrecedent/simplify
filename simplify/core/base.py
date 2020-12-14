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
        modules Union[str, Sequence[str]]: name(s) of module(s) where object to 
            load is/are located. Defaults to an empty list.                           
    """
    contents: Union[Callable, str] = None
    name: str = None
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    modules: Union[str, Sequence[str]] = dataclasses.field(
        default_factory = list)  
    
    """ Initialization Methods """
    
    def __post_init__(self) -> None:
        """Initializes class instance."""
        # Calls parent initialization methods, if they exist.
        try:
            super().__post_init__()
        except AttributeError:
            pass   
    
    """ Public Methods """
    
    def apply(self, data: object = None, **kwargs) -> object:
        """[summary]

        Args:
            data (object, optional): [description]. Defaults to None.

        Returns:
            object: [description]
        """
        self.algortihm = self.load(key = self.name)
        return super().apply(data = data, **kwargs)

             
@dataclasses.dataclass
class SimpleStep(sourdough.components.Step):
    """Wrapper for a SimpleTechnique.

    Subclasses of SimpleStep store additional methods and attributes to apply to 
    all possible technique instances that could be used. This is often useful 
    when using parallel Worklow instances which test a variety of strategies 
    with similar or identical parameters and/or methods.

    A SimpleStep instance will try to return attributes from SimpleTechnique if 
    the attribute is not found in the SimpleStep instance. 

    Args:
        contents (SimpleTechnique): technique instance to be used in a Workflow.
            Defaults ot None.
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Settings instance, 'name' 
            should match the appropriate section name in the Settings instance. 
            When subclassing, it is sometimes a good idea to use the same 'name' 
            attribute as the base class for effective coordination between 
            sourdough classes. 
                        
    """
    contents: Union[SimpleTechnique, str] = None
    name: str = None
                
    """ Properties """
    
    @property
    def technique(self) -> SimpleTechnique:
        return self.contents
    
    @technique.setter
    def technique(self, value: SimpleTechnique) -> None:
        self.contents = value
        return self
    
    @technique.deleter
    def technique(self) -> None:
        self.contents = None
        return self
    
    """ Public Methods """
    
    def apply(self, data: object = None, **kwargs) -> object:
        """Applies SimpleTechnique instance in 'contents'.
        
        The code below outlines a basic method that a subclass should build on
        for a properly functioning Step.
        
        Applies stored 'contents' with 'parameters'.
        
        Args:
            data (object): optional object to apply 'contents' to. Defaults to
                None.
                
        Returns:
            object: with any modifications made by 'contents'. If data is not
                passed, nothing is returned.        
        
        """
        if data is None:
            self.contents.apply(**kwargs)
            return self
        else:
            return self.contents.apply(data = data, **kwargs)

    """ Dunder Methods """

    def __getattr__(self, attribute: str) -> Any:
        """Looks for 'attribute' in 'contents'.

        Args:
            attribute (str): name of attribute to return.

        Raises:
            AttributeError: if 'attribute' is not found in 'contents'.

        Returns:
            Any: matching attribute.

        """
        try:
            return getattr(self.contents, attribute)
        except AttributeError:
            raise AttributeError(f'{attribute} neither found in {self.name} '
                                 f'nor {self.contents}') 
