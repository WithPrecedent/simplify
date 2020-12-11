"""
interface: user interface for siMpLify projects
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    SimpleBases (sourdough.Bases): base classes for a SimpleProject.
    SimpleProject (sourdough.Project): main access point and interface for
        creating and implementing data science projects.
    
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
class SimpleBases(sourdough.Bases):
    """Base classes for a siMpLify project.
    
    Args:
        settings (Type): the configuration class to use in a sourdough project.
            Defaults to sourdough.Settings.
        manager (Type): the file manager class to use in a sourdough project.
            Defaults to simplify.SimpleManager.   
        project (Type): the product/builder class to use in a sourdough 
            project. Defaults to simplify.SimpleCreator.    
        product (Type): the product output class to use in a sourdough 
            project. Defaults to sourdough.Product. 
        component (Type): the node class to use in a sourdough project. Defaults 
            to simplify.SimpleComponent. 
        workflow (Type): the workflow to use in a sourdough project. Defaults to 
            sourdough.products.Workflow.      
            
    """
    settings: Type = sourdough.Settings
    manager: Type = simplify.SimpleManager
    creator: Type = simplify.SimpleCreator
    product: Type = sourdough.Product
    component: Type = simplify.SimpleComponent
    workflow: Type = sourdough.products.Workflow
  

@dataclasses.dataclass
class SimpleProject(sourdough.Project):
    """Constructs, organizes, and implements a data science project.

    Args:
        contents (Mapping[str, object]]): stored objects created by the 
            'create' methods of 'creators'. Defaults to an empty dict.
        settings (Union[Type, str, pathlib.Path]]): a Settings-compatible class,
            a str or pathlib.Path containing the file path where a file of a 
            supported file type with settings for a Settings instance is 
            located. Defaults to the default Settings instance.
        manager (Union[Type, str, pathlib.Path]]): a Manager-compatible class,
            or a str or pathlib.Path containing the full path of where the root 
            folder should be located for file input and output. A 'manager'
            must contain all file path and import/export methods for use 
            throughout sourdough. Defaults to the default Manager instance. 
        creators (Sequence[Union[Type, str]]): a Creator-compatible classes or
            strings corresponding to the keys in registry of the default
            'creator' in 'bases'. Defaults to a list of 'simple_architect', 
            'simple_builder', and 'simple_worker'. 
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example if a 
            sourdough instance needs settings from a Settings instance, 'name' 
            should match the appropriate section name in the Settings instance. 
            When subclassing, it is sometimes a good idea to use the same 'name' 
            attribute as the base class for effective coordination between 
            sourdough classes. If it is None, the 'name' will be attempted to be 
            inferred from the first section name in 'settings' after 'general' 
            and 'files'. If that fails, 'name' will be the snakecase name of the
            class. Defaults to None. 
        identification (str): a unique identification name for a Project 
            instance. The name is used for creating file folders related to the 
            project. If it is None, a str will be created from 'name' and the 
            date and time. Defaults to None.   
        automatic (bool): whether to automatically advance 'director' (True) or 
            whether the director must be advanced manually (False). Defaults to 
            True.
        data (object): any data object for the project to be applied. If it is
            None, an instance will still execute its workflow, but it won't
            apply it to any external data. Defaults to None.  
        bases (ClassVar[object]): contains information about default base 
            classes used by a Project instance. Defaults to an instance of 
            SimpleBases.

    """
    contents: Sequence[Any] = dataclasses.field(default_factory = dict)
    settings: Union[object, Type, str, pathlib.Path] = None
    manager: Union[object, Type, str, pathlib.Path] = None
    creators: Sequence[Union[Type, str]] = dataclasses.field(
        default_factory = lambda: ['simple_architect', 'simple_builder', 
                                   'simple_worker'])
    name: str = None
    identification: str = None
    automatic: bool = True
    data: Union[pd.DataFrame, np.ndArray, simplify.Dataset] = None
    bases: ClassVar[object] = SimpleBases()
    options: sourdough.project.resources.Options = None
    rules: sourdough.project.resources.Rules = None
    system: ClassVar[SimpleSystem] = None

    """ Initialization Methods """
    
    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Calls validation methods based on items listed in 'validations'.
        for validation in sourdough.rules.validations:
            if ((self.system is not None and not self.system._validated)
                    or validation == 'data'):
                getattr(self, f'_validate_{validation}')()
        # Sets index for iteration.
        self.index = 0
        # Advances through 'creators' if 'automatic' is True.
        if self.automatic:
            self._auto_create()
        
    """ Private Methods """
    
    def _validate_data(self) -> None:
        """Validates 'data' or converts it to a Dataset instance."""
        pass


project_options = sourdough.types.Catalog(contents = {
    'wrangler': simplify.wrangler,
    'actuary': simplify.actuary, 
    'analyst': simplify.analyst, 
    'critic': simplify.critic, 
    'artist': simplify.artist})


@dataclasses.dataclass
class SimpleSystem(sourdough.types.Lexicon):
    """Constructs, organizes, and implements a data science project.

    Args:
        contents (Mapping[str, object]]): stored objects created by the 
            'create' methods of 'projects'. Defaults to an empty dict.
        settings (Union[Type, str, pathlib.Path]]): a Settings-compatible class,
            a str or pathlib.Path containing the file path where a file of a 
            supported file type with settings for a Settings instance is 
            located. Defaults to the default Settings instance.
        manager (Union[Type, str, pathlib.Path]]): a Manager-compatible class,
            or a str or pathlib.Path containing the full path of where the root 
            folder should be located for file input and output. A 'manager'
            must contain all file path and import/export methods for use 
            throughout sourdough. Defaults to the default Manager instance. 
        projects (Sequence[Union[Type, str]]): a Creator-compatible classes or
            strings corresponding to the keys in registry of the default
            'project' in 'bases'. Defaults to a list of 'simple_architect', 
            'simple_builder', and 'simple_worker'. 
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example if a 
            sourdough instance needs settings from a Settings instance, 'name' 
            should match the appropriate section name in the Settings instance. 
            When subclassing, it is sometimes a good idea to use the same 'name' 
            attribute as the base class for effective coordination between 
            sourdough classes. If it is None, the 'name' will be attempted to be 
            inferred from the first section name in 'settings' after 'general' 
            and 'files'. If that fails, 'name' will be the snakecase name of the
            class. Defaults to None. 
        identification (str): a unique identification name for a Project 
            instance. The name is used for creating file folders related to the 
            project. If it is None, a str will be created from 'name' and the 
            date and time. Defaults to None.   
        automatic (bool): whether to automatically advance 'director' (True) or 
            whether the director must be advanced manually (False). Defaults to 
            True.
        data (object): any data object for the project to be applied. If it is
            None, an instance will still execute its workflow, but it won't
            apply it to any external data. Defaults to None.  
        bases (ClassVar[object]): contains information about default base 
            classes used by a Project instance. Defaults to an instance of 
            SimpleBases.

    """
    contents: Mapping[str, SimpleProject] = dataclasses.field(
        default_factory = dict)
    settings: Union[object, Type, str, pathlib.Path] = None
    manager: Union[object, Type, str, pathlib.Path] = None
    projects: Mapping[str, ModuleType] = dataclasses.field(
        default_factory = lambda: project_options)
    name: str = None
    identification: str = None
    automatic: bool = True
    data: Union[pd.DataFrame, np.ndArray, simplify.Dataset] = None
    _validated: bool = False

    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        sourdough.rules.validations.append('data')
        self._validated = True
        # Calls parent and/or mixin initialization method(s).
        try:
            super().__post_init__()
        except AttributeError:
            pass
        
    """ Private Methods """

    def _create_project(self, project: str) -> SimpleProject:
        """[summary]

        Args:
            project (str): [description]

        Returns:
            SimpleProject: [description]
        """
        project = sourdough.tools.importify(
            module = self.projects[project],
            key = 'project')
        options = self._get_options(project = project)
        rules = self._get_rules(project = project, options = options)
        return project(
            settings = self.settings,
            manager = self.manager,
            identification = self.identification,
            data = self.data,
            system = self.system,
            options = options,
            rulse = rules)
     
    def _get_options(self, project: str) -> sourdough.project.resources.Options:
        """[summary]

        Args:
            project (str): [description]

        Returns:
            sourdough.project.resources.Options: [description]
        """
        options = getattr(self.projects[project], 'options')
        try:
            algorithms = getattr(self.projects[project], 'algorithms')
        except KeyError:
            algorithms = getattr(self.projects[project], 'get_algorithms')(
                settings = self.settings)
        options.algorithms = algorithms
        return options    
    
    def _get_rules(self, project: str, 
                   options: sourdough.project.resources.Options) -> (
                       sourdough.project.resources.Rules):
        """[summary]

        Args:
            project (str): [description]

        Returns:
            sourdough.project.resources.Rules: [description]
        """
        try:
            rules = getattr(self.projects[project], 'rules')
        except KeyError:
            rules = sourdough.rules
        rules.options = options
        return rules
           
    def _validate_data(self) -> None:
        """Validates 'data' or converts it to a Dataset instance."""
        pass

    
    def _auto_create(self) -> None:
        """Advances through the stored Creator instances.
        
        The results of the iteration is that each item produced is stored in 
        'content's with a key of the 'produces' attribute of each project.
        
        """
        for project in iter(self):
            self.contents.update({project.produces: self.__next__()})
        return self
    
    """ Dunder Methods """
    
    def __next__(self) -> Any:
        """Returns products of the next Creator in 'projects'.

        Returns:
            Any: item project by the 'create' method of a Creator.
            
        """
        if self.index < len(self.projects):
            project = self.projects[self.index]()
            if hasattr(self, 'verbose') and self.verbose:
                print(
                    f'{project.action} {project.produces} from {project.needs}')
            self.index += 1
            product = project.create(project = self)
        else:
            raise IndexError()
        return product
    
    def __iter__(self) -> Iterable:
        """Returns iterable of 'projects'.
        
        Returns:
            Iterable: iterable sequence of 'projects'.
            
        """
        return iter(self.projects)
      