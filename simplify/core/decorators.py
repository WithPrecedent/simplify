
from functools import wraps
from inspect import getfullargspec
import time
from types import FunctionType

from .tools import convert_time, listify

def check_df(method):
    """Decorator which automatically uses the default DataFrame if one
    is not passed to the decorated method.

    Parameters:
        method: wrapped method within a class instance.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        argspec = getfullargspec(method)
        unpassed_args = argspec.args[len(args):]
        if 'df' in argspec.args and 'df' in unpassed_args:
            kwargs.update({'df' : getattr(self, self.default_df)})
        return method(self, *args, **kwargs)

    return wrapper

def check_arguments(method, excludes = None):
    """Decorator which uses class instance attribute of the same name as a
    passed parameter if no argument is passed for that parameter and the
    parameter is not listed in excludes.

    Parameters:
        method: wrapped method within a class instance.
        excludes: list or string of parameters for which a local attribute
            should not be used.
    """
    if not excludes:
        excludes = []
    else:
        excludes = listify(excludes)

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        argspec = getfullargspec(method)
        unpassed_args = argspec.args[len(args):]
        for unpassed in unpassed_args:
            if unpassed not in excludes and hasattr(self, unpassed):
                kwargs.update({unpassed : getattr(self, unpassed)})
        return method(self, *args, **kwargs)

    return wrapper

def timer(process = None):
    """Decorator for computing the length of time a process takes.

    Parameters:
        process: string containing name of class or method."""

    if not process:
        if isinstance(process, FunctionType):
            process = process.__name__
        else:
            process = process.__class__.__name__

    def shell_timer(_function):

        def decorated(*args, **kwargs):
            start_time = time.time()
            result = _function(*args, **kwargs)
            total_time = time.time() - start_time
            h, m, s = convert_time(total_time)
            print(f'{process} completed in %d:%02d:%02d' % (h, m, s))
            return result

        return decorated

    return shell_timer
