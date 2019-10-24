"""
.. module:: builder
:synopsis: the builder design pattern made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections import namedtuple
from functools import partial
from importlib import import_module

"""
This module uses a traditional builder design pattern, with this module acting
as the defacto builder class, to create algorithms for use in the siMpLify
package. The director is any outside class or function which calls the builder
to create an algorithm and corresponding parameters.

Placing the builder functions here makes for more elegant code. To use these
functions, the importation line at the top of module using the builder just
needs:

    import builder

And then to create any new algorithm, the function calls are very
straightforward and clear. For example, to create a new algorithm and
corresponding parameters, this is all that is required:

    algorithm, parameters = builder.create(**builder_parameters)

Putting the builder in __init__.py takes advantage of the automatic importation
of the file when the folder is imported. As a result, the specific functions
do not need to be imported by name and/or located in another module that needs
to be imported.

Using the __init__.py file for builder functions was inspired by a blog post,
which used the file for a builder design pattern, by Luna at:
https://www.bnmetrics.com/blog/builder-pattern-in-python3-simple-version

"""

def create(configuration: dict, package: str, step: str, technique: str,
           parameters: dict = None, runtimes: dict = None):
    """Creates algorithm and corresponding parameters.

    Args:
        configuration (dict): settings dictionary from Idea instance.
        package (str): name of siMpLify subpackage being used.
        step (str): name of step in the subpackage being used.
        technique (str): name of the particular technique for which an
            algorithm and parameters are sought.
        parameters (dict, optional): any preexisting parameters. If this is
            passed, parameters from configuration will not be used. Defaults to
            None.
        runtimes (dict, optional): contains any parameters that can only be
            added at runtime. The keys should match the name of the parameter
            and the value should be a string matching the name of the attribute
            in the Technique namedtuple that is used to extract parameter
            information. Defaults to None.

    Returns:
        algorithm (object), parameters (dict): an algorithm object and
            corresponding, finalized parameters.

    """
    techniques_module = import_module(
        ''.join(['simplify.builder.', package, '_techniques']))
    technique = get_technique(
        configuration = configuration,
        techniques_module = techniques_module,
        step = step,
        technique = technique)
    algorithm = getattr(import_module(technique.module), technique.algorithm)
    parameters = get_parameters(
        configuration = configuration,
        step = step,
        technique = technique,
        parameters = parameters,
        runtimes = runtimes)
    return module.TechniqueWrapper(
        technique = technique.name,
        algorithm = algorithm,
        parameters = parameters,
        data_parameters = technique.data_parameters)

def get_technique(configuration: dict, techniques_module: str, step: str,
                  technique: str, gpu: bool = False):
    """[summary]

    Args:
        configuration (dict): [description]
        techniques_module (str): [description]
        step (str): [description]
        technique (str): [description]
        gpu (bool, optional): [description]. Defaults to False.
    """
    if configuration['general']['gpu']:
        gpu_string = 'gpu'
    else:
        gpu_string = ''
    try:
        return getattr(module, '_'.join([gpu_string, step, technique]))
    except AttributeError:
        return getattr(module, '_'.join([step, technique]))

def get_parameters(configuration: dict, step: str, technique: namedtuple,
                   parameters: dict, runtimes: dict = None):
    """[summary]

    Args:
        configuration (dict): [description]
        step (str): [description]
        technique (namedtuple): [description]
        parameters (dict): [description]
        runtimes (dict, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    parameters = get_configuration_parameters(
        configuration = configuration,
        step = step,
        technique = technique,
        parameters = parameters)
    parameters = get_selected_parameters(
        technique = technique,
        parameters = parameters)
    parameters = get_extra_parameters(
        technique = technique,
        parameters = parameters)
    parameters = get_runtime_parameters(
        configuration = configuration,
        step = step,
        technique = technique,
        parameters = parameters,
        runtimes = runtimes)
    parameters = get_conditional_parameters(
        technique = technique,
        parameters = parameters)
    return parameters

def get_configuration_parameters(configuration: dict, step: str,
                                 technique: namedtuple, parameters: dict):
    """

    Args:
        configuration (dict): [description]
        step (str): [description]
        technique (namedtuple): [description]
        parameters (dict): [description]

    Returns:
        [type]: [description]
    """
    if parameters:
        return parameters
    else:
        try:
            parameters = configuration[''.join([technique.name, '_parameters'])]
        except KeyError:
            try:
                parameters = configuration[''.join([step, '_parameters'])]
            except KeyError:
                pass
    return parameters

def get_selected_parameters(technique: namedtuple, parameters: dict):
    """[summary]

    Args:
        technique (namedtuple): [description]
        parameters (dict): [description]

    Returns:
        [type]: [description]
    """
    try:
        if technique.selected_parameters:
            parameters_to_use = list(
                technique.default_parameters.keys())
            new_parameters = {}
            for key, value in parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            parameters = new_parameters
    except AttributeError:
        pass
    return parameters

def get_extra_parameters(technique: namedtuple, parameters: dict):
    """[summary]

    Args:
        technique (namedtuple): [description]
        parameters (dict): [description]
    """
    try:
        parameters.update(technique.extra_parameters)
    except TypeError:
        pass
    return parameters

def get_runtime_parameters(configuration: dict, step: str,
                           technique: namedtuple, parameters: dict,
                           runtimes: dict):
    """[summary]

    Args:
        configuration (dict): [description]
        step (str): [description]
        technique (namedtuple): [description]
        parameters (dict): [description]
        runtimes (dict): [description]
    """
    try:
        for key, value in technique.runtime_parameters.items():
            try:
                parameters.update({key: runtimes[value]})
            except KeyError:
                try:
                    parameters.update({key: configuration['general'][value]})
                except KeyError:
                    try:
                        parameters.update({key: configuration[step][value]})
                    except KeyError:
                        error = 'no matching runtime parameter found'
                        raise KeyError(error)
    except AttributeError:
        pass
    return parameters

def get_conditional_parameters(technique: namedtuple, parameters: dict):
    """[summary]

    Args:
        technique (namedtuple): [description]
        parameters (dict): [description]
    """
    try:
        parameters = technique.conditional_parameters(
            technique = technique,
            parameters = parameters)
    except TypeError:
        pass
    return parameters

