"""    

Contents:

    timer (and related convert_time function): wrapper which can be attached
        to any class, function, or method to determine how long a process
        takes. Output is in hours, minutes, seconds, with the possibility
        of passing a string to the decorator so that printed output is linked
        to the operation or process.
    check_arguments: wrapper which checks for identically named local attributes
        if method arguments are unpassed or passed as None. A list of arguments
        not to substitute ('exclude') can be passed for wrapped methods so that
        certain designated arguments will not be replaced.
"""
from dataclasses import dataclass
from functools import wraps
from inspect import getfullargspec
import time
from types import FunctionType

def convert_time(seconds):
    """Function that converts seconds into hours, minutes, and seconds.
    
    Args:
        seconds: an int containing a nubmer of seconds.
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

def timer(process = None):
    """Decorator for computing the length of time a process takes.

    Args:
        process: string containing name of class or method to be used in the
            output describing time elapsed.
    """
    if not process:
        if isinstance(process, FunctionType):
            process = process.__name__
        else:
            process = process.__class__.__name__
    def shell_timer(_function):
        def decorated(*args, **kwargs):
            produce_time = time.time()
            result = _function(*args, **kwargs)
            total_time = time.time() - produce_time
            h, m, s = convert_time(total_time)
            print(f'{process} completed in %d:%02d:%02d' % (h, m, s))
            return result
        return decorated
    return shell_timer
    
def check_arguments(method, excludes = None):
    """Decorator which uses class instance attribute of the same name as a
    passed parameter if no argument is passed for that parameter and the
    parameter is not listed in excludes.

    Args:
        method: wrapped method within a class instance.
        excludes: list or string of parameters for which a local attribute
            should not be used.
    """
    if not excludes:
        excludes = []
    else:
        excludes = self.listify(excludes)

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        argspec = getfullargspec(method)
        unpassed_args = argspec.args[len(args):]
        for unpassed in unpassed_args:
            if unpassed not in excludes and hasattr(self, unpassed):
                kwargs.update({unpassed : getattr(self, unpassed)})
        return method(self, *args, **kwargs)

    return wrapper