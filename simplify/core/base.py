"""
base: core classes for a siMpLify data science project
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
class SimpleSettings(project.Settings):
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
    contents: Union[str, pathlib.Path, Mapping[str, Mapping[str, Any]]] = (
        dataclasses.field(default_factory = dict))
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
class SimpleFiler(project.Filer):
    
    pass
    
    
@dataclasses.dataclass
class SimpleProject(sourdough.Project):
    """Directs construction and execution of a data science project.
    
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a SimpleSettings instance, 
            'name' should match the appropriate section name in a SimpleSettings 
            instance. Defaults to None. 
        settings (Union[SimpleSimpleSettings, Type[SimpleSimpleSettings], 
            pathlib.Path, str, Mapping[str, Mapping[str, Any]]]): a 
            Settings-compatible subclass or instance, a str or pathlib.Path 
            containing the file path where a file of a supported file type with
            settings for a SimpleSettings instance is located, or a 2-level 
            mapping containing settings. Defaults to the default SimpleSettings 
            instance.
        filer (Union[SimpleSimpleFiler, Type[SimpleSimpleFiler], pathlib.Path, 
            str]): a SimpleFiler-compatible class or a str or pathlib.Path 
            containing the full path of where the root folder should be located 
            for file input and output. A 'filer' must contain all file path and 
            import/export methods for use throughout sourdough. Defaults to the 
            default SimpleFiler instance. 
        identification (str): a unique identification name for a sourdough
            Project. The name is used for creating file folders related to the 
            project. If it is None, a str will be created from 'name' and the 
            date and time. Defaults to None.   
        outline (project.Stage): an outline of a project workflow derived from 
            'settings'. Defaults to None.
        workflow (project.Stage): a workflow of a project derived from 
            'outline'. Defaults to None.
        summary (project.Stage): a summary of a project execution derived from 
            'workflow'. Defaults to None.
        automatic (bool): whether to automatically advance 'worker' (True) or 
            whether the worker must be advanced manually (False). Defaults to 
            True.
        data (Any): any data object for the project to be applied. If it is
            None, an instance will still execute its workflow, but it won't
            apply it to any external data. Defaults to None.  
        states (ClassVar[Sequence[Union[str, project.Stage]]]): a list of Stages 
            or strings corresponding to keys in 'bases.stage.library'. Defaults 
            to a list containing 'outline', 'workflow', and 'summary'.
        validations (ClassVar[Sequence[str]]): a list of attributes that need 
            validating. Defaults to a list of attributes in the dataclass field.
    
    Attributes:
        bases (ClassVar[sourdough.types.Lexicon]): a class attribute containing
            a dictionary of base classes with libraries of subclasses of those 
            bases classes. Changing this attribute will entirely replace the 
            existing links between this instance and all other base classes.
        
    """
    name: str = None
    settings: Union[SimpleSettings, Type[SimpleSettings], pathlib.Path, str, 
                    Mapping[str, Mapping[str, Any]]] = None
    filer: Union[SimpleFiler, Type[SimpleFiler], pathlib.Path, str] = None
    identification: str = None
    outline: SimpleStage = None
    workflow: SimpleStage = None
    summary: SimpleStage = None
    automatic: bool = True
    data: Union[str, np.ndarray, pd.DataFrame, pd.Series] = None
    stages: ClassVar[Sequence[Union[str, SimpleStage]]] = [
        'outline', 'workflow', 'summary']
    validations: ClassVar[Sequence[str]] = [
        'settings', 'name', 'identification', 'filer']
    

@dataclasses.dataclass
class SimpleStage(project.Stage):
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
class SimpleManager(project.Manager):
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
class SimpleComponent(project.Component):
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
     
                        
@dataclasses.dataclass
class SimpleTechnique(sourdough.quirks.Loader, SimpleComponent):
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
            