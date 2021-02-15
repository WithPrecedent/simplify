"""
stages: interim and final stages of a project
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    Outline
    Workflow
    Summary
    
"""
from __future__ import annotations
import copy
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import more_itertools

import sourdough
from . import base


@dataclasses.dataclass
class Outline(base.Stage):
    """Information needed to construct and execute a Workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration 
            instance, 'name' should match the appropriate section name in a 
            Configuration instance. Defaults to None. 
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

    """ Public Class Methods """
    
    @classmethod   
    def from_settings(cls, settings: base.Settings, 
                      name: str = None) -> Outline:
        """[summary]

        Args:
            source (base.Settings): [description]

        Returns:
            Outline: [description]
            
        """   
        skips = [k for k in settings.keys() if k.endswith(tuple(settings.skip))]
        component_keys = [k for k in settings.keys() if k not in skips]
        if name is None:
            try:
                name = component_keys[0]
            except IndexError:
                raise ValueError('No sections in settings indicate how to '
                                 'construct a project outline')
        structure = cls._get_structure(name = name, settings = settings) 
        outline = cls(name = name, structure = structure)      
        for section in component_keys:
            outline = cls._parse_section(name = section, 
                                         settings = settings,
                                         outline = outline)
        outline = cls._add_runtime_parameters(outline = outline, 
                                              settings = settings)
        return outline 
    
    """ Private Class Methods """
    
    @classmethod
    def _parse_section(cls, name: str, settings: base.Settings, 
                       outline: Outline) -> Outline:
        """[summary]

        Args:
            name (str): [description]
            settings (base.Settings): [description]
            outline (Outline): [description]

        Returns:
            Outline: [description]
        """        
        section = settings[name]
        design = cls._get_design(name = name, settings = settings)
        outline.designs[name] = design
        outline.initialization[name] = {}
        outline.attributes[name] = {}
        component = cls.bases.component.library.borrow(names = [name, design])
        parameters = tuple(i for i in list(component.__annotations__.keys()) 
                           if i not in ['name', 'contents'])
        for key, value in section.items():
            suffix = key.split('_')[-1]
            prefix = key[:-len(suffix) - 1]
            if suffix in ['design', 'workflow']:
                pass
            elif suffix in cls.bases.component.library.suffixes:
                outline.designs.update(dict.fromkeys(value, suffix[:-1]))
                outline.components[prefix] = value 
            elif suffix in parameters:
                outline.initialization[name][suffix] = value 
            elif prefix in [name]:
                outline.attributes[name][suffix] = value
            else:
                outline.attributes[name][key] = value
        return outline   

    @classmethod
    def _get_design(cls, name: str, settings: base.Settings) -> str:
        """[summary]

        Args:
            name (str): [description]
            settings (base.Settings):

        Raises:
            KeyError: [description]

        Returns:
            str: [description]
            
        """
        try:
            design = settings[name][f'{name}_design']
        except KeyError:
            try:
                design = settings[name][f'design']
            except KeyError:
                try:
                    design = settings['sourdough']['default_design']
                except KeyError:
                    raise KeyError(f'To designate a design, a key in settings '
                                   f'must either be named "design" or '
                                   f'"{name}_design"')
        return design    

    @classmethod
    def _get_structure(cls, name: str, settings: base.Settings) -> str:
        """[summary]

        Args:
            name (str): [description]
            section (Mapping[str, Any]): [description]

        Raises:
            KeyError: [description]

        Returns:
            str: [description]
            
        """
        try:
            structure = settings[name][f'{name}_workflow']
        except KeyError:
            try:
                structure = settings[name][f'workflow']
            except KeyError:
                try:
                    structure = settings['sourdough']['default_workflow']
                except KeyError:
                    raise KeyError(f'To designate a workflow structure, a key '
                                   f' in settings must either be named '
                                   f'"workflow" or "{name}_workflow"')
        return structure  

    @classmethod
    def _add_runtime_parameters(cls, outline: Outline, 
                                settings: base.Settings) -> Outline:
        """[summary]

        Args:
            outline (Outline): [description]
            settings (base.Settings): [description]

        Returns:
            Outline: [description]
            
        """
        for component in outline.components.keys():
            names = [component]
            if component in outline.designs:
                names.append(outline.designs[component])
            for name in names:
                try:
                    outline.runtime[name] = settings[f'{name}_parameters']
                except KeyError:
                    pass
        return outline
       

@dataclasses.dataclass
class Workflow(base.Stage, sourdough.Graph):
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
            a list with 'outline' and 'name'.
                   
    """
    contents: Dict[str, List[str]] = dataclasses.field(default_factory = dict)
    default: Any = dataclasses.field(default_factory = list)
    components: sourdough.Library = sourdough.Library()
    needs: ClassVar[Union[Sequence[str], str]] = ['outline', 'name']

    """ Public Class Methods """
            
    @classmethod
    def from_outline(cls, outline: Outline, name: str) -> Workflow:
        """Creates a Workflow from an Outline.

        Args:
            outline (Outline): [description]
            name (str): [description]

        Returns:
            Workflow: [description]
            
        """        
        workflow = cls()
        workflow = cls._add_component(name = name,
                                      outline = outline,
                                      workflow = workflow)
        for component in outline.components[name]:
            workflow = cls._add_component(name = component,
                                          outline = outline,
                                          workflow = workflow)
        return workflow
                             
    """ Public Methods """
    
    def combine(self, workflow: Workflow) -> None:
        """Adds 'other' Workflow to this Workflow.
        
        Combining creates an edge between every endpoint of this instance's
        Workflow and the every root of 'workflow'.
        
        Args:
            workflow (Workflow): a second Workflow to combine with this one.
            
        Raises:
            ValueError: if 'workflow' has nodes that are also in 'flow'.
            
        """
        if any(k in workflow.components.keys() for k in self.components.keys()):
            raise ValueError('Cannot combine Workflows with the same nodes')
        else:
            self.components.update(workflow.components)
        super().combine(structure = workflow)
        return self
   
    def execute(self, data: Any, copy_components: bool = True, **kwargs) -> Any:
        """Iterates over 'contents', using 'components'.
        
        Args:
            
        Returns:
            
        """
        for path in iter(self):
            data = self.execute_path(data = data, 
                                     path = path, 
                                     copy_components = copy_components, 
                                     **kwargs)  
        return data

    def execute_path(self, data: Any, path: Sequence[str], 
                     copy_components: bool = True, **kwargs) -> Any:
        """Iterates over 'contents', using 'components'.
        
        Args:
            
        Returns:
            
        """
        for node in more_itertools.always_iterable(path):
            if copy_components:
                component = copy.deepcopy(self.components[node])
            else:
                component = self.components[node]
            data = component.execute(data = data, **kwargs)    
        return data
            
    """ Private Class Methods """
    
    @classmethod
    def _add_component(cls, name: str, outline: Outline,
                       workflow: Workflow) -> Workflow:
        """[summary]

        Args:
            name (str): [description]
            details (Details): [description]
            workflow (Workflow): [description]

        Returns:
            Workflow: [description]
            
        """
        workflow.append(node = name)
        design = outline.designs[name]
        component = cls.bases.component.library.borrow(names = [name, design])
        instance = component.from_outline(name = name, outline = outline)
        workflow.components[name] = instance
        return workflow
  

@dataclasses.dataclass
class Summary(sourdough.types.Lexicon, base.Stage):
    """Collects and stores results of executing a Workflow.
    
    Args:
        contents (Mapping[Any, Any]]): stored dictionary. Defaults to an empty 
            dict.
        default (Any): default value to return when the 'get' method is used.
        prefix (str): prefix to use when storing different paths through a 
            workflow. So, for example, a prefix of 'path' will create keys of
            'path_1', 'path_2', etc. Defaults to 'path'.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to 
            a list with 'workflow' and 'data'.          
              
    """
    contents: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    default: Any = None
    prefix: str = 'path'
    needs: ClassVar[Union[Sequence[str], str]] = ['workflow', 'data']

    """ Public Methods """
    
    @classmethod
    def from_workflow(cls, workflow: Workflow, data: Any = None,
                      copy_data: bool = True, **kwargs) -> sourdough.Project:
        """[summary]

        Args:
            project (sourdough.Project): [description]

        Returns:
            sourdough.Project: [description]
            
        """
        summary = cls()
        for i, path in enumerate(workflow):
            key = f'{cls.prefix}_{str(i)}'
            if copy_data:
                to_use = copy.deepcopy(data)
            else:
                to_use = data
            summary.contents[key] = workflow.execute_path(data = to_use,
                                                          path = path,
                                                          **kwargs)
        return summary
        