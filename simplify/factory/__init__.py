"""
.. module:: factory
:synopsis: the factory design pattern made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from ..core.packages import SimpleComparer, SimpleSequencer
from ..core.plans import SimplePlan
from ..core.techniques import (FarmerTechnique, ChefTechnique, CriticTechnique,
                               ArtistTechnique)

"""
This module uses a traditional factory design pattern, with this module acting
as the defacto class, to create the primary class objects used in the siMpLify
packages.

Placing the factory functions here makes for more elegant code. To use these
functions, the importation line at the top of module using the factory just
needs:

    import factory

And then to create any new classes, the function calls are very straightforward
and clear. For example, to create a new SimpleTechnique, this is all that is
required:

    factory.create_technique(parameters)

Putting the factory in __init__.py takes advantage of the automatic importation
of the file when the folder is imported. As a result, the specific functions
do not need to be imported by name and/or located in another module that needs
to be imported.

Using the __init__.py file for factory functions was inspired by a blog post by
Luna at:
https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version

"""

def create(class_type: str, *args, **kwargs):
    """Calls appropriate function based upon 'class_type' passed.

    This method adds nothing to calling the functions directly. It is simply
    included for easier external iteration and/or for those who prefer a generic
    function for all factories.

    Args:
        class_type(str): name of package type to be created. It should
            correspond to the suffix of one of the other functions in this
            module (following the prefix 'create_').
        *args, **kwargs: appropriate arguments to be passed to corresponding
            factory function.

    Returns:
        instance of new class created by the factory.

    Raises:
        TypeError: if there is no corresponding function for creating a class
            instance designated in 'class_type'.

    """
    if class_type in get_supported_types():
        return locals()['create_' + class_type](*args, **kwargs)
    else:
        error = class_type + ' is not a valid class type'
        raise TypeError(error)

def create_package(order: list, comparer: object = None,
                   comparer_iterable: str = None, draft_method: object = None,
                   publish_method: object = None,
                   instance_parameters: dict = {}):
    """Creates package or package instance.

    If comparer or comparer_iterable is passed, the final class or class
    instance returned will be a SimpleComparer. Otherwise, a class or class
    instance of SimpleSequencer will be returned.

    Args:
        order(list(str)): step names in order.
        comparer(SimplePlan): SimplePlan subclass to contain a set of steps.
        comparer_iterable(str): the name of the local attribute for storing
            instances of the 'comparer'.
        draft_method(func): a replacement 'draft' method if the default method
            is insufficient.
        publish_method(func): a replacement 'publish' method if the default
            method is insufficient.
        instance_parameters(dict): parameters to be passed to an instance of
            the new class. This parameter is optional. If it is not passed, a
            class, and not a class instance, will be returned.

    Returns:
        package(SimplePackage): class or class instance of a SimplePackage
            subclass.

    """
    if comparer or comparer_iterable:
        package = SimpleComparer()
        package.comparer = comparer
        package.comparer_iterable = comparer_iterable
    else:
        package = SimpleSequencer()
    package.order = order
    if draft_method:
        package.draft = draft_method
    if publish_method:
        package.publish = publish_method
    if instance_parameters:
        package = package(**instance_parameters)
    return package

def create_plan(draft_method: object = None, publish_method: object = None,
                instance_parameters: dict = {}):
    """Creates plan or plan instance.

    Args:
        draft_method(func): a replacement 'draft' method if the default method
            is insufficient.
        publish_method(func): a replacement 'publish' method if the default
            method is insufficient.
        instance_parameters(dict): parameters to be passed to an instance of
            the new class. This parameter is optional. If it is not passed, a
            class, and not a class instance, will be returned.

    Returns:
        plan(SimplePlan): class or class instance of a SimplePlan subclass.

    """
    plan = SimplePlan()
    if draft_method:
        plan.draft = draft_method
    if publish_method:
        plan.publish = publish_method
    if instance_parameters:
        plan = plan(**instance_parameters)
    return plan

def create_technique(technique_type: str, draft_method: object = None,
                     publish_method: object = None,
                     instance_parameters: dict = {}):
    """Creates technique or technique instance.

    Args:
        technique_type(str): name corresponding to the name of the technique
            class to be created. Supported names are stored in
            'technique_classes'.
        draft_method(func): a replacement 'draft' method if the default method
            is insufficient.
        publish_method(func): a replacement 'publish' method if the default
            method is insufficient.
        instance_parameters(dict): parameters to be passed to an instance of
            the new class. This parameter is optional. If it is not passed, a
            class, and not a class instance, will be returned.

    Returns:
        techinique(SimpleTechnique): class or class instance of a
            SimpleTechnique subclass.

    """
    technique_classes = {
        'farmer': FarmerTechnique,
        'chef': ChefTechnique,
        'critic': CriticTechnique,
        'artist': ArtistTechnique}
    technique = technique_classes[technique_type]
    if draft_method:
        technique.draft = draft_method
    if publish_method:
        technique.publish = publish_method
    if instance_parameters:
        technique = technique(**instance_parameters)
    return technique

def get_supported_types():
    """Removes 'create_' from object names in locals() to create a list of
    supported class types.

    Returns:
        class_types(list): class types which have a corresponding factory
            function for creation.

    """
    class_types = []
    for factory_function in locals().keys():
        if factory_function.startswith('create_'):
            class_types.append(factory_function[len('create_'):])
    return class_types
