"""
analyst.split
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0) 

Contents:

"""
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import sourdough

from . import base
import simplify


@dataclasses.dataclass
class Split(sourdough.project.Step):
    """Wrapper for a Technique.

    An instance will try to return attributes from 'contents' if the attribute 
    is not found in the Step instance. 

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a Configuration instance, 
            'name' should match the appropriate section name in a Configuration 
            instance. Defaults to None.
        contents (Technique): stored Technique instance used by the 'implement' 
            method.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Component design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            True.
                                                
    """    
    name: str = 'split'
    contents: sourdough.project.Technique = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    parallel: ClassVar[bool] = True


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
        module (str): name of module where 'contents' is located.
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

splitters = sourdough.types.Library(
    contents = {
        'train_test_split': SklearnSplitter(
            name = 'train test',
            contents = 'ShuffleSplit',
            parameters = base.Parameters(
                name = 'train_test',
                default = {'n_splits': 1, 'test_size': 0.33, 'shuffle': True}, 
                runtime = {'random_state': 'seed'}),
            module = 'sklearn.model_selection'),
        'kfold_split': SklearnSplitter(
            name = 'kfold',
            contents = 'Kfold',
            parameters = base.Parameters(
                name = 'kfold',
                default = {'n_splits': 5, 'shuffle': True},  
                runtime = {'random_state': 'seed'}),
            module = 'sklearn.model_selection'),
        'stratified_kfold_split': SklearnSplitter(
            name = 'stratified kfold',
            contents = 'Stratified_KFold',
            parameters = base.Parameters(
                name = 'stratified_kfold',
                default = {'n_splits': 5, 'shuffle': True},  
                runtime = {'random_state': 'seed'}),
            module = 'sklearn.model_selection'),
        'group_kfold_split': SklearnSplitter(
            name = 'group kfold',
            contents = 'GroupKFold',
            parameters = base.Parameters(
                name = 'group_kfold',
                default = {'n_splits': 5, 'shuffle': True},  
                runtime = {'random_state': 'seed'}),
            module = 'sklearn.model_selection'),
        'time_series_split': SklearnSplitter(
            name = 'time series split',
            contents = 'Group_KFold',
            parameters = base.Parameters(
                name = 'time_series_split',
                default = {'n_splits': 5, 'shuffle': True},  
                runtime = {'random_state': 'seed'}),
            module = 'sklearn.model_selection')})
