"""
components: core components of a data science workflow
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    
"""
from __future__ import annotations
import abc
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

from . import base
import sourdough


@dataclasses.dataclass
class SimpleStep(base.SimpleComponent):
    """Wrapper for a Technique.

    Subclasses of Step can store additional methods and attributes to implement
    all possible technique instances that could be used. This is often useful 
    when using parallel Worklow instances which test a variety of strategies 
    with similar or identical parameters and/or methods.

    A Step instance will try to return attributes from Technique if the
    attribute is not found in the Step instance. 

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout siMpLify. For example, if a 
            siMpLify instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
            instance. Defaults to None.
        contents (Technique): stored Technique instance used by the 'implement' 
            method.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            False.
                                                
    """
    name: str = None
    contents: SimpleTechnique = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    parallel: ClassVar[bool] = False

                     
@dataclasses.dataclass
class SimpleTechnique(sourdough.quirks.Importer, base.SimpleComponent):
    """Base class for primitive objects in a siMpLify workflow.
    
    The 'contents' and 'parameters' attributes are combined at the last moment
    to allow for runtime alterations.
    
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout siMpLify. For example, if a 
            siMpLify instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
            instance. Defaults to None.
        contents (Any): stored item for use by a Component subclass instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            False.
                                               
    """

    name: str = None
    contents: base.SimpleAlgorithm = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    parallel: ClassVar[bool] = False
    module: str = None
    default: Dict[str, Any] = dataclasses.field(default_factory = dict)

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
            