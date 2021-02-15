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
import copy
import dataclasses
import inspect
import pathlib
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import more_itertools
import sourdough

from . import quirks


@dataclasses.dataclass
class SimpleBases(object):
    """Stores base classes in siMpLify.
     
    """
    
    def register(self, name: str, item: Union[Type, object]) -> None:
        """[summary]
        Args:
            name (str): [description]
            item (Union[Type, object]): [description]
        Raises:
            ValueError: [description]
            TypeError: [description]
        Returns:
            [type]: [description]
            
        """
        if name in dir(self):
            raise ValueError(f'{name} is already registered')
        elif inspect.isclass(item) and issubclass(item, SimpleBase):
            setattr(self, name, item)
        elif isinstance(item, SimpleBase):
            setattr(self, name, item.__class__)
        else:
            raise TypeError(f'item must be a SimpleBase')
        return self

    def remove(self, name: str) -> None:
        """[summary]
        Args:
            name (str): [description]
        Raises:
            AttributeError: [description]
            
        """
        try:
            delattr(self, name)
        except AttributeError:
            raise AttributeError(f'{name} does not exist in {self.__name__}')


@dataclasses.dataclass
class SimpleBase(abc.ABC):
    """Base mixin for automatic registration of subclasses and instances. 
    
    Any concrete (non-abstract) subclass will automatically store itself in the 
    class attribute 'subclasses' using the snakecase name of the class as the 
    key.
    
    Any direct subclass will automatically store itself in the class attribute 
    'bases' using the snakecase name of the class as the key.
    
    Any instance of a subclass will be stored in the class attribute 'instances'
    as long as '__post_init__' is called (either by a 'super()' call or if the
    instance is a dataclass and '__post_init__' is not overridden).
    
    Args:
        bases (ClassVar[SimpleBases]): library that stores direct subclasses 
            (those with Base in their '__bases__' attribute) and allows runtime 
            access and instancing of those stored subclasses.
    
    Attributes:
        subclasses (ClassVar[sourdough.types.Catalog]): library that stores 
            concrete subclasses and allows runtime access and instancing of 
            those stored subclasses. 'subclasses' is automatically created when 
            a direct SimpleBase subclass (SimpleBase is in its '__bases__') is 
            instanced.
        instances (ClassVar[sourdough.types.Catalog]): library that stores
            subclass instances and allows runtime access of those stored 
            subclass instances. 'instances' is automatically created when a 
            direct SimpleBase subclass (SimpleBase is in its '__bases__') is 
            instanced. 
                      
    Namespaces: 
        bases, subclasses, instances, borrow, instance, and __init_subclass__.
    
    """
    bases: ClassVar[SimpleBases] = SimpleBases()
    
    """ Initialization Methods """
    
    def __init_subclass__(cls, **kwargs):
        """Adds 'cls' to appropriate class libraries."""
        super().__init_subclass__(**kwargs)
        # Creates a snakecase key of the class name.
        key = sourdough.tools.snakify(cls.__name__)
        # Adds class to 'bases' if it is a base class.
        if SimpleBase in cls.__bases__:
            # Creates libraries on this class base for storing subclasses.
            cls.subclasses = sourdough.types.Catalog()
            cls.instances = sourdough.types.Catalog()
            # Adds this class to 'bases' using 'key'.
            cls.bases.register(name = key, item = cls)
        # Adds concrete subclasses to 'library' using 'key'.
        if not abc.ABC in cls.__bases__:
            cls.subclasses[key] = cls

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Calls parent and/or mixin initialization method(s).
        try:
            super().__post_init__()
        except AttributeError:
            pass
        try:
            key = self.name
        except AttributeError:
            key = sourdough.tools.snakify(self.__class__.__name__)
        self.instances[key] = self
 
    """ Public Class Methods """
    
    @classmethod
    def borrow(cls, name: Union[str, Sequence[str]]) -> Type[SimpleBase]:
        """[summary]
        Args:
            name (Union[str, Sequence[str]]): [description]
        Raises:
            KeyError: [description]
        Returns:
            SimpleBase: [description]
            
        """
        item = None
        for key in more_itertools.always_iterable(name):
            try:
                item = cls.subclasses[key]
                break
            except KeyError:
                pass
        if item is None:
            raise KeyError(f'No matching item for {str(name)} was found') 
        else:
            return item
           
    @classmethod
    def instance(cls, name: Union[str, Sequence[str]], **kwargs) -> SimpleBase:
        """[summary]
        Args:
            name (Union[str, Sequence[str]]): [description]
        Raises:
            KeyError: [description]
        Returns:
            SimpleBase: [description]
            
        """
        item = None
        for key in more_itertools.always_iterable(name):
            for library in ['instances', 'subclasses']:
                try:
                    item = getattr(cls, library)[key]
                    break
                except KeyError:
                    pass
            if item is not None:
                break
        if item is None:
            raise KeyError(f'No matching item for {str(name)} was found') 
        elif inspect.isclass(item):
            return cls(name = name, **kwargs)
        else:
            instance = copy.deepcopy(item)
            for key, value in kwargs.items():
                setattr(instance, key, value)
            return instance


@dataclasses.dataclass
class Component(SimpleBase, sourdough.quirks.Element, abc.ABC):
    """Base class for parts of a sourdough Workflow.
    
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout siMpLify. For example, if a siMpLify 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
                
    Attributes:
        bases (ClassVar[SimpleBases]): library that stores siMpLify base classes 
            and allows runtime access and instancing of those stored subclasses.
        subclasses (ClassVar[sourdough.types.Catalog]): library that stores 
            concrete subclasses and allows runtime access and instancing of 
            those stored subclasses. 
        instances (ClassVar[sourdough.types.Catalog]): library that stores
            subclass instances and allows runtime access of those stored 
            subclass instances.
                
    """
    name: str = None

    """ Required Subclass Methods """
    
    @abc.abstractmethod
    def execute(self, project: sourdough.Project, 
                **kwargs) -> sourdough.Project:
        """[summary]
        Args:
            project (sourdough.Project): [description]
        Returns:
            sourdough.Project: [description]
            
        """ 
        return project

    @abc.abstractmethod
    def implement(self, project: sourdough.Project, 
                  **kwargs) -> sourdough.Project:
        """[summary]
        Args:
            project (sourdough.Project): [description]
        Returns:
            sourdough.Project: [description]
            
        """  
        return project
        
    """ Public Class Methods """
    
    @classmethod
    def create(cls, name: Union[str, Sequence[str]], **kwargs) -> Component:
        """[summary]
        Args:
            name (Union[str, Sequence[str]]): [description]
        Raises:
            KeyError: [description]
        Returns:
            Component: [description]
            
        """        
        keys = more_itertools.always_iterable(name)
        for key in keys:
            for library in ['instances', 'subclasses']:
                item = None
                try:
                    item = getattr(cls, library)[key]
                    break
                except KeyError:
                    pass
            if item is not None:
                break
        if item is None:
            raise KeyError(f'No matching item for {str(name)} was found') 
        elif inspect.isclass(item):
            return cls(name = name, **kwargs)
        else:
            instance = copy.deepcopy(item)
            for key, value in kwargs.items():
                setattr(instance, key, value)
            return instance


@dataclasses.dataclass
class Stage(SimpleBase, sourdough.quirks.Needy, abc.ABC):
    """Creates a sourdough object.
    
    Args:
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to an
            empty list.     
                
    Attributes:
        bases (ClassVar[SimpleBases]): library that stores siMpLify base classes 
            and allows runtime access and instancing of those stored subclasses.
        subclasses (ClassVar[sourdough.types.Catalog]): library that stores 
            concrete subclasses and allows runtime access and instancing of 
            those stored subclasses. 
        instances (ClassVar[sourdough.types.Catalog]): library that stores
            subclass instances and allows runtime access of those stored 
            subclass instances.
                       
    """
    needs: ClassVar[Union[Sequence[str], str]] = []
