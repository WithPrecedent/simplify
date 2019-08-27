from functools import wraps
from inspect import getfullargspec, signature

import time


def check_df(func):
    """Decorator which automatically uses the default DataFrame if one
    is not passed to the decorated method.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        argspec = getfullargspec(func)
        unpassed_args = argspec.args[len(args):]
        if 'df' in argspec.args and 'df' in unpassed_args:
            kwargs.update({'df' : getattr(self, self.default_df)})
        return func(self, *args, **kwargs)
    return wrapper

def check_kwargs(method, excludes = None):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
#        argspec = getfullargspec(method)
#        unpassed_args = argspec.args[len(args):]
#        sig = dict(signature(method).bind(variables).arguments)
#        if not excludes:
#            excludes = []
#        for unpassed in unpassed_args:
#            if not in excludes and hasattr(self, unpassed):
#
#
#        if 'recipe' in unpassed_args:
#            for variable in variables:
#                if variable in argspec.args and test_var in unpassed_args:
#                    kwargs.update({variable : getattr(self, variable)})
#        elif 'recipe' in argspec.args:
#            kwargs.update({'recipe' : sig['recipe']})
#            x, y = sig['recipe'].ingredients[self.data_to_plot]
#            if 'x' in argspec.args and 'x' in unpassed_args:
#                kwargs.update({'x' : x})
#            if 'y' in argspec.args and 'y' in unpassed_args:
#                kwargs.update({'y' : y})
#            if ('predicted_probs' in argspec.args
#                    and 'predicted_probs' in unpassed_args):
#                new_param = getattr(sig['recipe'],
#                                    'evalutor.predicted_probs')
#                kwargs.update({'predicted_probs' : new_param})
#            if ('estimator' in argspec.args
#                    and 'predicted_probs' in unpassed_args):
#                new_param = getattr(sig['recipe'], 'model.algorithm')
#                kwargs.update({'estimator' : new_param})
        return method(self, *args, **kwargs)
    return wrapper


def convert_time(seconds):
    """Function that converts seconds into hours, minutes, and seconds."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

def timer(process):
    """Decorator for computing the length of time a process takes."""
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
