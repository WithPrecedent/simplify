"""
analyst.scale
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0) 

Contents:

"""
import abc
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import numpy as np
import pandas as pd
import sourdough

from . import base
import simplify


@dataclasses.dataclass
class Scale(sourdough.project.Step):
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
    name: str = 'scale'
    contents: sourdough.project.Technique = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    parallel: ClassVar[bool] = True
    

@dataclasses.dataclass
class MinMaxScale(simplify.components.SklearnTransformer):
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
    name: str = 'min_max'
    contents: Union[Callable, Type, object, str] = 'MinMaxScaler'
    iterations: Union[int, str] = 1
    parameters: Union[Mapping[str, Any], 
                      base.Parameters] = base.Parameters(
                          name = 'min_max_scale',
                          default = {'copy': False}, 
                          selected = ['copy'])
    module: str = 'sklearn.preprocessing'
    parallel: ClassVar[bool] = False



scalers = sourdough.types.Library()


@dataclasses.dataclass
class MaxAbsoluteScale(simplify.components.SklearnTransformer):
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
    name: str = 'max_absolute'
    contents: Union[Callable, Type, object, str] = 'MaxAbsScaler'
    iterations: Union[int, str] = 1
    parameters: Union[Mapping[str, Any], 
                      base.Parameters] = base.Parameters(
                          name = 'max_absolute_scale',
                          default = {'copy': False}, 
                          selected = ['copy'])
    module: str = 'sklearn.preprocessing'
    parallel: ClassVar[bool] = False


@dataclasses.dataclass
class NormalizeScale(simplify.components.SklearnTransformer):
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
    name: str = 'normalize'
    contents: Union[Callable, Type, object, str] = 'Normalizer'
    iterations: Union[int, str] = 1
    parameters: Union[Mapping[str, Any], 
                      base.Parameters] = base.Parameters(
                          name = 'normalize_scale',
                          default = {'copy': False}, 
                          selected = ['copy'])
    module: str = 'sklearn.preprocessing'
    parallel: ClassVar[bool] = False


@dataclasses.dataclass
class QuantileScale(simplify.components.SklearnTransformer):
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
    name: str = 'quantile'
    contents: Union[Callable, Type, object, str] = 'QuantileTransformer'
    iterations: Union[int, str] = 1
    parameters: Union[Mapping[str, Any], 
                      base.Parameters] = base.Parameters(
                          name = 'quantile_scale',
                          default = {'copy': False}, 
                          selected = ['copy'])
    module: str = 'sklearn.preprocessing'
    parallel: ClassVar[bool] = False


@dataclasses.dataclass
class RobustScale(simplify.components.SklearnTransformer):
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
    name: str = 'robust'
    contents: Union[Callable, Type, object, str] = 'RobustScaler'
    iterations: Union[int, str] = 1
    parameters: Union[Mapping[str, Any], 
                      base.Parameters] = base.Parameters(
                          name = 'robust_scale',
                          default = {'copy': False}, 
                          selected = ['copy'])
    module: str = 'sklearn.preprocessing'
    parallel: ClassVar[bool] = False


@dataclasses.dataclass
class StandardScale(simplify.components.SklearnTransformer):
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
    name: str = 'standard'
    contents: Union[Callable, Type, object, str] = 'StandardScaler'
    iterations: Union[int, str] = 1
    parameters: Union[Mapping[str, Any], 
                      base.Parameters] = base.Parameters(
                          name = 'standard_scale',
                          default = {'copy': False}, 
                          selected = ['copy'])
    module: str = 'sklearn.preprocessing'
    parallel: ClassVar[bool] = False


# @dataclasses.dataclass
# class GaussScale(simplify.components.SklearnTransformer):
#     """Transforms data columns to more gaussian distribution.

#     The particular method applied is chosen between 'box-cox' and 'yeo-johnson'
#     based on whether the particular data column has values below zero.

#     Args:
#         step(str): name of step used.
#         parameters(dict): dictionary of parameters to pass to selected
#             algorithm.
#         name(str): name of class for matching settings in the Idea instance
#             and for labeling the columns in files exported by Critic.
#         auto_draft(bool): whether 'finalize' method should be called when
#             the class is instanced. This should generally be set to True.
#     """
#     name: str = 'box-cox & yeo-johnson'
#     contents: str = None
#     iterations: Union[int, str] = 1
#     parameters: Union[Mapping[str, Any]. 
#                       base.Parameters] = base.Parameters(
#                           name = 'gauss_scale',
#                           default = {'rescaler': 'standard'})
#     module: str = None
#     parallel: ClassVar[bool] = False
    

#     def __post_init__(self) -> None:
#         self.idea_sections = ['analyst']
#         super().__post_init__()
#         return self

#     def draft(self) -> None:
#         self.rescaler = self.parameters['rescaler'](
#                 copy = self.parameters['copy'])
#         del self.parameters['rescaler']
#         self._publish_parameters()
#         self.positive_tool = self.workers['box_cox'](
#                 method = 'box_cox', **self.parameters)
#         self.negative_tool = self.workers['yeo_johnson'](
#                 method = 'yeo_johnson', **self.parameters)
#         return self

#     def publish(self, dataset, columns = None):
#         if not columns:
#             columns = dataset.numerics
#         for column in columns:
#             if dataset.x[column].min() >= 0:
#                 dataset.x[column] = self.positive_tool.fit_transform(
#                         dataset.x[column])
#             else:
#                 dataset.x[column] = self.negative_tool.fit_transform(
#                         dataset.x[column])
#             dataset.x[column] = self.rescaler.fit_transform(
#                     dataset.x[column])
#         return dataset
