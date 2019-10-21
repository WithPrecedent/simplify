"""
.. module:: builder
:synopsis: the builder design pattern made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from functools import partial

from simplify.builder.parameters import get_parameters



"""
This module uses a traditional builder design pattern, with this module acting
as the defacto class, to create algorithms for use in the siMpLify package.

Placing the builder functions here makes for more elegant code. To use these
functions, the importation line at the top of module using the builder just
needs:

    import builder

And then to create any new algorithm, the function calls are very
straightforward and clear. For example, to create a new algorithm derived from
the scikit-learn package, this is all that is required:

    builder.create_sklearn(parameters)

Putting the builder in __init__.py takes advantage of the automatic importation
of the file when the folder is imported. As a result, the specific functions
do not need to be imported by name and/or located in another module that needs
to be imported.

Using the __init__.py file for builder functions was inspired by a blog post,
which used the file for a builder design pattern, by Luna at:
https://www.bnmetrics.com/blog/builder-pattern-in-python3-simple-version

"""
technique = namedtuple('technique',
    ['algorithm', 'parameters', 'default_parameters', 'extra_parameters',
     'runtime_parameters', 'search_parameters', 'data'])

def create(algorithm_type: str, technique: namedtuple):
    """Calls appropriate function based upon 'algorithm_type' passed.

    This method adds nothing to calling the functions directly. It is simply
    included for easier external iteration and/or for those who prefer a generic
    function for all builders.

    Args:
        algorithm_type(str): name of algorithm type to be created. It should
            correspond to the suffix of one of the other functions in this
            module (following the prefix 'create_').
        *args, **kwargs: appropriate arguments to be passed to corresponding
            builder function.

    Returns:
        instance of new class created by the builder.

    Raises:
        TypeError: if there is no corresponding function for creating a class
            instance designated in 'algorithm_type'.

    """
    if algorithm_type in get_supported_types():
        return locals()['create_' + algorithm_type](technique = technique)
    else:
        error = algorithm_type + ' is not a valid class type'
        raise TypeError(error)


def simplify_alogorithm(technique: namedtuple):
    """Creates algorithm or algorithm instance.

    Args:
        algorithm_type(str): name corresponding to the name of the algorithm
            class to be created. Supported names are stored in
            'algorithm_classes'.
        draft_method(func): a replacement 'draft' method if the default method
            is insufficient.
        publish_method(func): a replacement 'publish' method if the default
            method is insufficient.
        instance_parameters(dict): parameters to be passed to an instance of
            the new class. This parameter is optional. If it is not passed, a
            class, and not a class instance, will be returned.

    Returns:
        techinique(Simplealgorithm): class or class instance of a
            Simplealgorithm subclass.

    """
    algorithm_classes = {
        'farmer': Farmeralgorithm,
        'chef': Chefalgorithm,
        'critic': Criticalgorithm,
        'artist': Artistalgorithm}
    algorithm = algorithm_classes[algorithm_type]
    if draft_method:
        algorithm.draft = draft_method
    if publish_method:
        algorithm.publish = publish_method
    if instance_parameters:
        algorithm = algorithm(**instance_parameters)
    return algorithm

def sklearn_algorithm(technique: namedtuple):
    """Creates algorithm or algorithm instance.

    Args:
        algorithm_type(str): name corresponding to the name of the algorithm
            class to be created. Supported names are stored in
            'algorithm_classes'.
        draft_method(func): a replacement 'draft' method if the default method
            is insufficient.
        publish_method(func): a replacement 'publish' method if the default
            method is insufficient.
        instance_parameters(dict): parameters to be passed to an instance of
            the new class. This parameter is optional. If it is not passed, a
            class, and not a class instance, will be returned.

    Returns:
        techinique(Simplealgorithm): class or class instance of a
            Simplealgorithm subclass.

    """
    algorithm_classes = {
        'farmer': Farmeralgorithm,
        'chef': Chefalgorithm,
        'critic': Criticalgorithm,
        'artist': Artistalgorithm}
    algorithm = algorithm_classes[algorithm_type]
    if draft_method:
        algorithm.draft = draft_method
    if publish_method:
        algorithm.publish = publish_method
    if instance_parameters:
        algorithm = algorithm(**instance_parameters)
    return algorithm


def get_supported_types():
    """Removes 'create_' from object names in locals() to create a list of
    supported class types.

    Returns:
        algorithm_types(list): class types which have a corresponding builder
            function for creation.

    """
    algorithm_types = []
    for builder_function in locals().keys():
        if builder_function.startswith('create_'):
            algorithm_types.append(builder_function[len('create_'):])
    return algorithm_types
