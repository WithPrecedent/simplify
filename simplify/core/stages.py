"""
stages: stages of construction in a siMpLify data science project
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    SimpleSettings (sourdough.project.Settings):
    SimpleFiler (sourdough.project.Filer):
    SimpleProject (sourdough.Project): main access point and interface for
        creating and implementing data science projects.
    SimpleManager (sourdough.project.Manager):
    SimpleComponent (sourdough.project.Component):
    SimpleOutline (sourdough.project.Outline):
    SimpleWorkflow (sourdough.project.Workflow):
    SimpleSummary (sourdough.project.Summary):
    
"""
from __future__ import annotations
import abc
import dataclasses
import pathlib
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import numpy as np
import pandas as pd

import simplify
import sourdough
from sourdough import project


@dataclasses.dataclass
class SimpleOutline(project.Outline):
    """Information needed to construct and execute a Workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a SimpleSettings instance, 
            'name' should match the appropriate section name in a SimpleSettings 
            instance. Defaults to None. 
        structure (str): the name matching the type of workflow to be used in a
            project. Defaults to None.
        components (Dict[str, List]): a dictionary with keys that are names of
            components and values that are lists of subcomponents for the keys. 
            Defaults to an empty dict.
        designs (Dict[str, str]): a dictionary with keys that are names of 
            components and values that are the names of the design structure for
            the keys. Defaults to an empty dict.
        initialization (Dict[str, Dict[str, Any]]): a dictionary with keys that 
            are the names of components and values which are dictionaries of 
            pararmeters to use when created the component listed in the key. 
            Defaults to an empty dict.
        runtime (Dict[str, Dict[str, Any]]): a dictionary with keys that 
            are the names of components and values which are dictionaries of 
            pararmeters to use when calling the 'execute' method of the 
            component listed in the key. Defaults to an empty dict.
        attributes (Dict[str, Dict[str, Any]]): a dictionary with keys that 
            are the names of components and values which are dictionaries of 
            attributes to automatically add to the component constructed from
            that key. Defaults to an empty dict.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to a
            list with 'settings' and 'name'.
            
    """
    name: str = None
    structure: str = None
    components: Dict[str, List] = dataclasses.field(default_factory = dict)
    designs: Dict[str, str] = dataclasses.field(default_factory = dict)
    initialization: Dict[str, Dict[str, Any]] = dataclasses.field(
        default_factory = dict)
    runtime: Dict[str, Dict[str, Any]] = dataclasses.field(
        default_factory = dict)
    attributes: Dict[str, Dict[str, Any]] = dataclasses.field(
        default_factory = dict)
    needs: ClassVar[Union[Sequence[str], str]] = ['settings', 'name']

       
@dataclasses.dataclass
class Workflow(project.Workflow):
    """Stores lightweight workflow and corresponding components.
    
    Args:
        contents (Dict[str, List[str]]): an adjacency list where the keys are 
            the names of nodes and the values are names of nodes which the key 
            is connected to. Defaults to an empty dict.
        default (Any): default value to use when a key is missing and a new
            one is automatically corrected. Defaults to an empty list.
        components (sourdough.Library): stores Component instances that 
            correspond to nodes in 'contents'. Defaults to an empty Library.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to 
            a list with 'outline' and 'name'
                   
    """
    contents: Dict[str, List[str]] = dataclasses.field(default_factory = dict)
    default: Any = dataclasses.field(default_factory = list)
    components: sourdough.Library = sourdough.Library()
    needs: ClassVar[Union[Sequence[str], str]] = ['outline', 'name']
    
    
@dataclasses.dataclass
class SimpleSummary(project.Summary):
    """Collects and stores results of executing a data science project workflow.
    
    Args:
        contents (Mapping[Any, Any]]): stored dictionary. Defaults to an empty 
            dict.
        default (Any): default value to return when the 'get' method is used.
        prefix (str): prefix to use when storing different paths through a 
            workflow. So, for example, a prefix of 'path' will create keys of
            'path_1', 'path_2', etc. Defaults to 'experiment'.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to 
            a list with 'workflow' and 'data'.          
              
    """
    contents: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    default: Any = None
    prefix: str = 'experiment'
    needs: ClassVar[Union[Sequence[str], str]] = ['workflow', 'data'] 

                        
@dataclasses.dataclass
class SimpleStep(sourdough.project.Step):
    """Wrapper for a SimpleTechnique.
    
    Subclasses can store additional methods and attributes to implement
    all possible technique instances that could be used. This is often useful 
    when using parallel Worklow instances which test a variety of strategies 
    with similar or identical parameters and/or methods.

    An instance will try to return attributes from SimpleTechnique if the
    attribute is not found in the SimpleStep instance. 
    
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
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
    contents: Callable = None
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)

                   
@dataclasses.dataclass
class SimpleTechnique(sourdough.sourdough.quirks.Importer, SimpleComponent):
    """Base class for primitive objects in a siMpLify workflow.
    
    The 'contents' and 'parameters' attributes are combined at the last moment
    to allow for runtime alterations.
    
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
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
    contents: Callable = None
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
            