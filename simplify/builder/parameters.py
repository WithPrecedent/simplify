"""
.. module:: parameters
:synopsis: aggregates, selects, and finalizes parameters for siMpLify builder
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

PARAMETER_TYPES= {
    'parameters': 'parameters',
    'selected': 'selected_parameters',
    'runtime': 'runtime_parameters',
    'extra': 'extra_parameters',
    'conditional': 'conditional_parameters'}

def get_algorithm_parameters(instance, parameter_types = None):
    """Returns parameters from different possible sources based upon
    instance attributes.

    Args:
        instance(object): class instance for parameters to be determined
        parameter_types(list(str)):  names of parameter groups in the instance
            class. If not provided, all of the keys in PARAMETER_TYPES will be
            incorporated if they exist.

    Returns
        parameters(dict): a finalized dictionary of parameters.

    """
    parameters = {}
    # Sets which groupings of parameters to incorporate.
    if not parameter_types:
        parameter_types = list(PARAMETER_TYPES.keys())
    # Iterates through possible parameter groups and adjusts 'parameters'
    for key, value in PARAMETER_TYPES.items():
        if key in parameter_types:
            # Looks for matching method in instance first.
            try:
                parameters = getattr(instance, '_get_' + value)(
                    parameters = parameters)
            except AttributeError:
                # Looks for instance dict with 'technique' as a key.
                try:
                    parameters.update(
                        getattr(instance, value)[instance.technique])
                except KeyError or AttributeError:
                    # Looks for attribute in instance.
                    try:
                        parameters.update(getattr(instance, value))
                    except AttributeError:
                        # Otherwise uses function from this module.
                        try:
                            parameters = locals()['get_' + value](
                                instance = instance,
                                parameters = parameters)
                        except KeyError:
                            pass
    return parameters

def get_parameters(instance, parameters):
    """Returns initial or default parameters to be processed from instance.

    Args:
        instance(object): subclass for parameters to be added.

    Returns
        parameters(dict): an initialized dictionary of parameters.

    """
    if not instance.parameters:
        # Tries to use 'default_parameters' if no parameters exist.
        try:
            parameters = instance.default_parameters[instance.technique]
        except KeyError or AttributeError:
            try:
                parameters = instance.default_parameters
            except AttributeError:
                pass
    return parameters

def get_selected_parameters(instance, parameters):
    """For classes that only need a subset of the parameters, this function
    selects that subset.

    Args:
        parameters_to_use(list): list or string containing names of
            parameters to include in final parameters dict.
    """
    new_parameters = {}
    parameters_to_use = []
    try:
        parameters_to_use = list(
            instance.default_parameters[instance.technique].keys())
    except KeyError or AttributeError:
        try:
            parameters_to_use = list(instance.default_parameters.keys())
        except AttributeError:
            pass
    if parameters_to_use:
        for key, value in parameters.items():
            if key in parameters_to_use:
                new_parameters.update({key: value})
        parameters = new_parameters
    return parameters

