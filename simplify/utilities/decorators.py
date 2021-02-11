"""
.. module:: decorators
:synopsis: sourdough decorators
:author: Corey Rayburn Yung
:copyright: 2020
:license: Apache-2.0
"""
from __future__ import annotations
import datetime
import inspect
import functools
import time
import types
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import sourdough


def namify(process: Callable) -> Callable:
    """Adds 'name' attribute to 'process' if none is passed.
    
    Args:
        process (Callable"""
    @functools.wraps(process)
    def wrapped(*args, **kwargs):
        call_signature = inspect.signature(process)
        arguments = dict(call_signature.bind(*args, **kwargs).arguments)
        if not arguments.get('name'):
            try:
                name = sourdough.tools.snakify(process.__name__)
            except AttributeError:
                name = sourdough.tools.snakify(process.__class__.__name__)
            arguments['name'] = name
        return process(**arguments)
    return wrapped
    
def timer(process: str = None) -> Callable:
    """Decorator for computing the length of time a process takes.

    Args:
        process (str): name of class or method to be used in the
            output describing time elapsed.

    """
    if not process:
        if isinstance(process, types.FunctionType):
            process = process.__class__.__name__
        else:
            process = process.__class__.__name__
    def shell_timer(_function):
        def decorated(*args, **kwargs):
            def convert_time(seconds: int) -> tuple(int, int, int):
                minutes, seconds = divmod(seconds, 60)
                hours, minutes = divmod(minutes, 60)
                return hours, minutes, seconds
            implement_time = time.time()
            result = _function(*args, **kwargs)
            total_time = time.time() - implement_time
            h, m, s = convert_time(total_time)
            print(f'{process} completed in %d:%02d:%02d' % (h, m, s))
            return result
        return decorated
    return shell_timer
