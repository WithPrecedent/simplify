"""
.. module:: factory
:synopsis: the factory pattern made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from ..core.package import SimpleComparer, SimpleSequencer
from ..core.plans import SimplePlan
from ..core.technique import SimpleTechnique

"""
Using the __init__.py file for factory functions was inspired by a blog post by 
Luna at:
https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version

Placing the factory functions here makes for my elegant code. The importation
line at the top of module using the factory just needs:

import factory

And then to create any new classes, the function calls are very straightforward
and clear. For example, to create a new SimpleTechnique, this is all that is
required:

factory.create_technique(parameters)

Putting the factory in __init__.py takes advantage of the automatic importation
of the file when the folder is imported. As a result, the specific functions
do not need to be imported by name and/or located in another module that needs
to be imported.

"""

def create(class_type, *args, **kwargs):
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

def create_package(name: str, options: dict, steps: dict, plans: dict, 
                   comparer: object, comparer_iterable :str):
    if plans or comparer or comparer_iterable:
        return SimpleComparer(
            name = name,
            options = options,
            steps = steps,
            plans = plans,
            comparer = comparer,
            comparer_iterable = comparer_iterable,
            auto_draft = auto_draft,
            auto_publish = auto_publish)
    else:
        return SimpleSequencer(
            name = name,
            options = options,
            steps = steps,
            auto_draft = auto_draft,
            auto_publish = auto_publish)

def create_plan():
    plan = SimplePlan()
    return plan

def create_technique():
    technique = SimpleTechnique()
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
