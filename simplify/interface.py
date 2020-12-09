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
        creator (Type): the product/builder class to use in a sourdough 
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


@dataclasses.dataclass
class SimpleSystem(sourdough.Project):
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
    contents: Mapping[str, SimpleProject] = dataclasses.field(
        default_factory = dict)
    settings: Union[object, Type, str, pathlib.Path] = None
    manager: Union[object, Type, str, pathlib.Path] = None
    creators: Sequence[Union[Type, str]] = dataclasses.field(
        default_factory = lambda: ['simple_architect', 'simple_builder', 
                                   'simple_worker'])
    name: str = None
    identification: str = None
    automatic: bool = True
    data: Union[pd.DataFrame, np.ndArray, simplify.Dataset] = None
    _validated: bool = False
    bases: ClassVar[object] = SimpleBases()

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
    
    def _validate_data(self) -> None:
        """Validates 'data' or converts it to a Dataset instance."""
        pass
  