"""
base: core classes for a siMpLify data science project
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    SimpleSettings (sourdough.project.Settings):
    SimpleFiler (sourdough.project.Filer):
    SimpleManager (sourdough.project.Manager):
    SimpleComponent (sourdough.project.Component):
    SimpleAlgorithm
    SimpleCriteria
    
    
    
"""
from __future__ import annotations
import abc
import dataclasses
import pathlib
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

from . import quirks
import sourdough
from sourdough import project


@dataclasses.dataclass
class SimpleSettings(quirks.SimpleBase, project.Settings):
    """Loads and stores configuration settings for a Project.

    Args:
        contents (Union[str, pathlib.Path, Mapping[str, Mapping[str, Any]]]): a 
            dict, a str file path to a file with settings, or a pathlib Path to
            a file with settings. Defaults to en empty dict.
        infer_types (bool): whether values in 'contents' are converted to other 
            datatypes (True) or left alone (False). If 'contents' was imported 
            from an .ini file, a False value will leave all values as strings. 
            Defaults to True.
        defaults (Mapping[str, Mapping[str]]): any default options that should
            be used when a user does not provide the corresponding options in 
            their configuration settings. Defaults to a dict with 'general', 
            'files', and 'sourdough' sections listed.
        skip (Sequence[str]): names of suffixes to skip when constructing nodes
            for a sourdough project. Defaults to a list with 'general', 'files',
            'sourdough', and 'parameters'.
        library (ClassVar[Library]): related Library instance that will store
            subclasses and allow runtime construction and instancing of those
            stored subclasses.    
            
    """
    contents: Union[Mapping[str, Mapping[str, Any]],
                    pathlib.Path,
                    str] = (dataclasses.field(default_factory = dict))
    infer_types: bool = True
    defaults: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory = lambda: {'general': {'verbose': False,
                                               'parallelize': False,
                                               'conserve_memery': False},
                                   'files': {'source_format': 'csv',
                                             'interim_format': 'csv',
                                             'final_format': 'csv',
                                             'file_encoding': 'windows-1252'},
                                   'sourdough': {'default_design': 'pipeline',
                                                 'default_workflow': 'graph'}})
    skip: Sequence[str] = dataclasses.field(
        default_factory = lambda: ['general', 
                                   'files', 
                                   'sourdough', 
                                   'parameters'])
    library: ClassVar[sourdough.Library] = sourdough.Library()


@dataclasses.dataclass
class SimpleFiler(quirks.SimpleBase, project.Filer):
    
    pass
    
  
@dataclasses.dataclass
class SimpleStage(quirks.SimpleBase, project.Stage):
    """Creates a siMpLify object.

    Args:
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to an
            empty list.
        library (ClassVar[Library]): related Library instance that will store
            subclasses and allow runtime construction and instancing of those
            stored subclasses.              
            
    """
    needs: ClassVar[Union[Sequence[str], str]] = []
    library: ClassVar[sourdough.Library] = sourdough.Library()  


@dataclasses.dataclass
class SimpleManager(quirks.SimpleBase, project.Manager):
    """Manages a distinct portion of a data science project workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a SimpleSettings instance, 
            'name' should match the appropriate section name in a SimpleSettings 
            instance. Defaults to None. 
        workflow (sourdough.Structure): a workflow of a project subpart derived 
            from 'outline'. Defaults to None.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to an
            empty list.
        library (ClassVar[Library]): related Library instance that will store
            subclasses and allow runtime construction and instancing of those
            stored subclasses.
                
    """
    name: str = None
    workflow: sourdough.Structure = None
    needs: ClassVar[Union[Sequence[str], str]] = ['outline', 'name']
    library: ClassVar[sourdough.Library] = sourdough.Library()


 @dataclasses.dataclass
class SimpleComponent(quirks.SimpleBase, project.Component):
    """Base class for parts of a data science project workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a SimpleSettings instance, 
            'name' should match the appropriate section name in a SimpleSettings 
            instance. Defaults to None. 
        contents (Any): stored item(s) for use by a Component subclass instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            False.
        library (ClassVar[Library]): related Library instance that will store
            subclasses and allow runtime construction and instancing of those
            stored subclasses.    
                
    """
    name: str = None
    contents: Any = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    parallel: ClassVar[bool] = False
    library: ClassVar[sourdough.Library] = sourdough.Library()
    

@dataclasses.dataclass
class SimpleAlgorithm(quirks.SimpleBase, sourdough.quirks.Importer):
    
    name: str = None
    module: str = None
    contents: str = None
    required: str = None


@dataclasses.dataclass
class SimpleCriteria(SimpleAlgorithm):
    
    name: str = None
    module: str = None
    contents: str = None
    required: str = None
