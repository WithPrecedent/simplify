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
class Parameters(sourdough.types.Lexicon):
    """Creates and stores parameters for a siMpLify component.
    
    Args:
        name (str):
        contents (Mapping[str, Any]):
        default: ClassVar[Mapping[str, str]] = {}
        runtime: ClassVar[Mapping[str, str]] = {}
        required: ClassVar[Sequence[str]] = []
        selected: ClassVar[Sequence[str]] = []   

    """
    name: str = None
    contents: Mapping[str, Any] = dataclasses.field(default_factory = dict)
    default: ClassVar[Mapping[str, str]] = {}
    runtime: ClassVar[Mapping[str, str]] = {}
    required: ClassVar[Sequence[str]] = []
    selected: ClassVar[Sequence[str]] = []
      
    """ Public Methods """

    def finalize(self, project: sourdough.Project, **kwargs) -> None:
        """[summary]

        Args:
            name (str):
            project (sourdough.Project):
            
        """
        # Uses kwargs or 'default' parameters as a starting base.
        self.contents = kwargs if kwargs else self.default
        # Adds any parameters from 'settings'.
        self.contents.update(self._get_from_settings(
            settings = project.settings))
        # Adds any required parameters.
        for item in self.required:
            if item not in self.contents:
                self.contents[item] = self.default[item]
        # Adds any runtime parameters.
        if self.runtime:
            self.add_runtime(project = project) 
            # Limits parameters to those selected.
            if self.selected:
                self.contents = {k: self.contents[k] for k in self.selected}
        return self

    """ Private Methods """
    
    def _add_runtime(self, project: sourdough.Project, **kwargs) -> None:
        """[summary]

        Args:
            project (sourdough.Project):
            
        """    
        for parameter, attribute in self.runtime.items():
            try:
                self.contents[parameter] = getattr(project, attribute)
            except AttributeError:
                try:
                    self.contents[parameter] = project.contents[attribute]
                except (KeyError, AttributeError):
                    pass
        if self.selected:
            self.contents = {k: self.contents[k] for k in self.selected}
        return self
     
    def _get_from_settings(self, settings: Mapping[str, Any]) -> Dict[str, Any]: 
        """[summary]

        Args:
            name (str): [description]
            settings (Mapping[str, Any]): [description]

        Returns:
            Dict[str, Any]: [description]
            
        """
        try:
            parameters = settings[f'{self.name}_parameters']
        except KeyError:
            suffix = self.name.split('_')[-1]
            prefix = self.name[:-len(suffix) - 1]
            try:
                parameters = settings[f'{prefix}_parameters']
            except KeyError:
                try:
                    parameters = settings[f'{suffix}_parameters']
                except KeyError:
                    parameters = {}
        return parameters


@dataclasses.dataclass
class SimpleProcess(base.Component, abc.ABC):
    """Base class for parts of a sourdough Workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
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
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    parallel: ClassVar[bool] = False
    
    """ Public Methods """
    
    def execute(self, project: sourdough.Project, 
                **kwargs) -> sourdough.Project:
        """[summary]

        Args:
            project (sourdough.Project): [description]

        Returns:
            sourdough.Project: [description]
            
        """ 
        if self.iterations in ['infinite']:
            while True:
                project = self.implement(project = project, **kwargs)
        else:
            for iteration in range(self.iterations):
                project = self.implement(project = project, **kwargs)
        return project

    def implement(self, project: sourdough.Project, 
                  **kwargs) -> sourdough.Project:
        """[summary]

        Args:
            project (sourdough.Project): [description]

        Returns:
            sourdough.Project: [description]
            
        """  
        if self.parameters:
            parameters = self.parameters
            parameters.update(kwargs)
        else:
            parameters = kwargs
        if self.contents not in [None, 'None', 'none']:
            project = self.contents.execute(project = project, **parameters)
        return project


@dataclasses.dataclass
class SimpleStep(SimpleProcess):
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
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    parallel: ClassVar[bool] = False

    
@dataclasses.dataclass
class SimpleTechnique(SimpleProcess):
    """Wrapper for a Technique.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
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
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    module: str = None
    parallel: ClassVar[bool] = False   

              
@dataclasses.dataclass
class Worker(SimpleProcess):
    """Base class for parts of a sourdough Workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
            instance. Defaults to None.
        contents (Any): stored item(s) for use by a Component subclass instance.
        workflow (sourdough.Structure): a workflow of a project subpart derived 
            from 'outline'. Defaults to None.
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
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    parallel: ClassVar[bool] = False
    
    """ Public Class Methods """

    @classmethod
    def from_outline(cls, name: str,
                     outline: sourdough.project.Outline, **kwargs) -> Worker:
        """[summary]

        Args:
            name (str): [description]
            outline (sourdough.project.Outline): [description]

        Returns:
            Worker: [description]
            
        """        
        worker = super().from_outline(name = name, outline = outline, **kwargs)
        if hasattr(worker, 'workflow'):
            worker.workflow = cls.bases.stage.library.borrow(
                names = 'workflow')()
            if worker.parallel:
                method = cls._create_parallel
            else:
                method = cls._create_serial
            worker = method(worker = worker, outline = outline)
        return worker
                  
    """ Private Class Methods """ 

    @classmethod                
    def _create_parallel(cls, worker: Worker,
                         outline: sourdough.project.Outline) -> Worker:
        """[summary]

        Args:
            worker (Worker): [description]
            outline (sourdough.project.Outline): [description]

        Returns:
        
        """
        name = worker.name
        step_names = outline.components[name]
        possible = [outline.components[s] for s in step_names]
        worker.workflow.branchify(nodes = possible)
        for i, step_options in enumerate(possible):
            for option in step_options:
                technique = cls.from_outline(name = option, outline = outline)
                wrapper = cls.from_outline(name = step_names[i],
                                           outline = outline,
                                           contents = technique)
                worker.workflow.components[option] = wrapper
        return worker 
    
    @classmethod
    def _create_serial(cls, worker: Worker,
                       outline: sourdough.project.Outline) -> Worker:                     
        """[summary]

        Args:
            worker (Worker): [description]
            outline (sourdough.project.Outline): [description]

        Returns:
        
        """
        print('test serial', worker.name)
        name = worker.name
        components = cls._depth_first(name = name, outline = outline)
        collapsed = list(more_itertools.collapse(components))
        worker.workflow.extend(nodes = collapsed)
        for item in collapsed:
            component = cls.from_outline(name = item, outline = outline)
            worker.workflow.components[item] = component
        return worker

    @classmethod
    def _depth_first(cls, name: str, outline: base.Stage) -> List:
        """

        Args:
            name (str):
            details (Blueprint): [description]

        Returns:
            List[List[str]]: [description]
            
        """
        organized = []
        components = outline.components[name]
        for item in components:
            organized.append(item)
            if item in outline.components:
                organized_subcomponents = []
                subcomponents = cls._depth_first(name = item, outline = outline)
                organized_subcomponents.append(subcomponents)
                if len(organized_subcomponents) == 1:
                    organized.append(organized_subcomponents[0])
                else:
                    organized.append(organized_subcomponents)
        return organized
     

@dataclasses.dataclass
class Pipeline(Worker):
    """
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
            instance. Defaults to None.
        contents (Callable): stored item used by the 'implement' method.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            True.    
                        
    """
    name: str = None
    contents: Any = None
    workflow: sourdough.Structure = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    parallel: ClassVar[bool] = False
    

@dataclasses.dataclass
class ParallelWorker(Worker, abc.ABC):
    """Resolves a parallel workflow by selecting the best option.

    It resolves a parallel workflow based upon criteria in 'contents'
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
            instance. Defaults to None.
        contents (Callable): stored item used by the 'implement' method.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            True.    
                        
    """
    name: str = None
    contents: Any = None
    workflow: sourdough.Structure = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    criteria: Callable = None
    parallel: ClassVar[bool] = True

    """ Public Methods """
    
    def implement(self, data: Any, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
            
        """        
        if hasattr(data, 'parallelize') and data.parallelize:
            method = self._implement_in_parallel
        else:
            method = self._implement_in_serial
        return method(data = data, **kwargs)

    """ Private Methods """
   
    def _implement_in_parallel(self, data: Any, **kwargs) -> Any:
        """Applies 'implementation' to 'project' using multiple cores.

        Args:
            project (Project): sourdough project to apply changes to and/or
                gather needed data from.
                
        Returns:
            Project: with possible alterations made.       
        
        """
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as pool:
            data = pool.starmap(self._implement_in_serial, data, **kwargs)
        return data 

    def _implement_in_serial(self, data: Any, **kwargs) -> Any:
        """Applies 'implementation' to 'project' using multiple cores.

        Args:
            project (Project): sourdough project to apply changes to and/or
                gather needed data from.
                
        Returns:
            Project: with possible alterations made.       
        
        """
        for path in self.workflow.permutations:
            data = self._implement_path(data = data, path = path, **kwargs)
        return data
    
    def _implement_path(self, data: Any, path: List[str], **kwargs) -> Any:  
        for node in path:
            component = self.workflow.components[node]
            data = component.execute(data = data, **kwargs)
        return data
    
       
@dataclasses.dataclass
class Contest(ParallelWorker):
    """Resolves a parallel workflow by selecting the best option.

    It resolves a parallel workflow based upon criteria in 'contents'
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
            instance. Defaults to None.
        contents (Callable): stored item used by the 'implement' method.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            True.    
                        
    """
    name: str = None
    contents: Any = None
    workflow: sourdough.Structure = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    criteria: Callable = None
    parallel: ClassVar[bool] = True

    """ Public Methods """
    
    def implement(self, data: Any, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
        """
                
        return data   
 
    
@dataclasses.dataclass
class Study(ParallelWorker):
    """Allows parallel workflow to continue

    A Study might be wholly passive or implement some reporting or alterations
    to all parallel workflows.
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
            instance. Defaults to None.
        contents (Callable): stored item used by the 'implement' method.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            True.   
                         
    """
    name: str = None
    contents: Any = None
    workflow: sourdough.Structure = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    criteria: Callable = None
    parallel: ClassVar[bool] = True

    """ Public Methods """
    
    def implement(self, data: Any, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
        """           
        return data    

    
@dataclasses.dataclass
class Survey(ParallelWorker):
    """Resolves a parallel workflow by averaging.

    It resolves a parallel workflow based upon the averaging criteria in 
    'contents'
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
            instance. Defaults to None.
        contents (Callable): stored item used by the 'implement' method.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            True.   
                          
    """
    name: str = None
    contents: Any = None
    workflow: sourdough.Structure = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    parallel: ClassVar[bool] = True

    """ Public Methods """
    
    def implement(self, data: Any, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
        """           
        return data   
    
    
@dataclasses.dataclass
class SklearnModel(sourdough.project.Technique):
    """Wrapper for a scikit-learn model (an algorithm that doesn't transform).

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
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
    contents: Union[Callable, Type, object, str] = None
    iterations: Union[int, str] = 1
    parameters: Union[Mapping[str, Any], 
                      base.Parameters] = base.Parameters()
    module: str = None
    parallel: ClassVar[bool] = False
    
    """ Public Methods """
    
    def implement(self, project: sourdough.Project) -> sourdough.Project:
        """[summary]

        Args:
            project (sourdough.Project): [description]

        Returns:
            sourdough.Project: [description]
            
        """
        try:
            self.parameters = self.parameters.finalize(project = project)
        except AttributeError:
            pass
        self.contents = self.contents(**self.parameters)
        self.contents.fit[project.data.x_train]
        return project


@dataclasses.dataclass
class SklearnSplitter(sourdough.project.Technique):
    """Wrapper for a scikit-learn data splitter.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
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
    contents: Union[Callable, Type, object, str] = None
    iterations: Union[int, str] = 1
    parameters: Union[Mapping[str, Any], 
                      base.Parameters] = base.Parameters()
    module: str = None
    parallel: ClassVar[bool] = False

    """ Public Methods """
    
    def implement(self, project: sourdough.Project) -> sourdough.Project:
        """[summary]

        Args:
            project (sourdough.Project): [description]

        Returns:
            sourdough.Project: [description]
            
        """
        try:
            self.parameters = self.parameters.finalize(project = project)
        except AttributeError:
            pass
        self.contents = self.contents(**self.parameters)
        project.data.splits = tuple(self.contents.split(project.data.x))
        project.data.split()
        return project
    
    
@dataclasses.dataclass
class SklearnTransformer(sourdough.project.Technique):
    """Wrapper for a scikit-learn transformer.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
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
    contents: Union[Callable, Type, object, str] = None
    iterations: Union[int, str] = 1
    parameters: Union[Mapping[str, Any], 
                      base.Parameters] = base.Parameters()
    module: str = None
    parallel: ClassVar[bool] = False
    
    """ Public Methods """
    
    def implement(self, project: sourdough.Project) -> sourdough.Project:
        """[summary]

        Args:
            project (sourdough.Project): [description]

        Returns:
            sourdough.Project: [description]
            
        """
        try:
            self.parameters = self.parameters.finalize(project = project)
        except AttributeError:
            pass
        self.contents = self.contents(**self.parameters)
        data = project.data
        data.x_train = self.contents.fit[data.x_train]
        data.x_train = self.contents.transform(data.x_train)
        if data.x_test is not None:
            data.x_test = self.contents.transform(data.x_test)
        if data.x_validate is not None:
            data.x_validate = self.contents.transform(data.x_validate)
        project.data = data
        return project
               