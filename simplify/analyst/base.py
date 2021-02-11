"""
analyst.base:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    Analyst (SimpleManager):
    Report (SimpleSummary):
    
"""
from __future__ import annotations
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import simplify


@dataclasses.dataclass
class Analyst(simplify.base.SimpleManager):
    """Manages a distinct portion of a data science project workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a SimpleSettings
            instance, 'name' should match the appropriate section name in a 
            SimpleSettings instance. Defaults to None. 
        workflow (simplify.base.SimpleWorkflow): a workflow of a project subpart 
            derived from 'outline'. Defaults to None.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to an
            empty list.
                
    """
    name: str = 'analyst'
    workflow: simplify.base.SimpleWorkflow = None
    needs: ClassVar[Union[Sequence[str], str]] = ['outline', 'name']


@dataclasses.dataclass
class Report(simplify.base.SimpleSummary):
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
    