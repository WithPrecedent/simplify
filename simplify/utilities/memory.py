"""
.. module:: memory
:synopsis: memory conservation utilities
:author: Corey Rayburn Yung
:copyright: 2020
:license: Apache-2.0
"""
from __future__ import annotations
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)


def add_slots(cls) -> object:
    """Adds slots to dataclass with default values.
    
    Derived from code here: 
    https://gitquirks.com/ericvsmith/dataclasses/blob/master/dataclass_tools.py
    
    Args:
        cls: class to add slots to
        
    Raises:
        TypeError: if '__slots__' is already in cls.
        
    Returns:
        object: class with '__slots__' added.
        
    """
    if '__slots__' in cls.__dict__:
        raise TypeError(f'{cls.__name__} already contains __slots__')
    else:
        cls_dict = dict(cls.__dict__)
        field_names = tuple(f.name for f in dataclasses.field(cls))
        cls_dict['__slots__'] = field_names
        for field_name in field_names:
            cls_dict.pop(field_name, None)
        cls_dict.pop('__dict__', None)
        qualname = getattr(cls, '__qualname__', None)
        cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
        if qualname is not None:
            cls.__qualname__ = qualname
    return cls
