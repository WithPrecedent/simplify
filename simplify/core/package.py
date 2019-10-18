"""
.. module:: iterable
:synopsis: iterable builder and container
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.base import SimpleClass


@dataclass
class SimplePackage(SimpleClass):
    """Parent class for building and controlling iterable steps.

    This class adds methods useful to create iterators, iterate over user
    options, and transform data or fit models. SimplePackage classes define
    the bulk of the siMpLify processing packages (Farmer, Chef, Critic, Artist)
    with only the lowest layer of the class hierarchy in a project being
    SimpleTechnique subclasses.

    To take maximum advantage of this class's functionality, a subclass should
    either, in its draft method, call super().draft() and/or define the
    following attributes there:
        order_setting(str): name of key in an Idea instance where the
            options for the iterable are listed. The names of the values
            corresponding to that key should be keys in the local 'options'
            dictionary.
        return_variables(dict(str: list) or list)): indicates which, if any,
            attributes should be incorporated into the local class from one or
            more of the classes stored in the class's iterable values. If the
            'return_variables' is a list, then the same attributes will be
            incorporated for each class instance in the iterable. If it is a
            nested dictionary, the outer_keys should correspond to keys in the
            local 'options' dictionary and the values should be lists of
            variables to return specific to option.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    steps: object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """
    
    def _check_package(self):
        if not hasattr(self, 'return_variables'):
            self.return_variables = []
        if not hasattr(self, 'order_setting'):
            self.order_setting = self.name + '_steps'
        if not hasattr(self, 'order'):
            self.order = []
        if not hasattr(self, 'steps'):
            self.steps = {}    
        return self

    def _infuse_return_variables(self, instance, return_variables = None,
                                 return_as_dict = False):
        """Adds 'return_variables' attributes from instance class to present
        class.

        Args:
            instance(object): class instance with attributes to be added to the
                present subclass instance.
            return_variables(list(str) or dict(str: list(str))): names of
                attributes sought. If stored in a dict, the outer key
                corresponds to particular keys stored in 'options'.

        """
        add_prefix = False
        if return_variables is None and self.exists('return_variables'):
            if isinstance(self.return_variables, dict):
                if instance.name in self.return_variables:
                    return_variables = self.return_variables[instance.name]
                else:
                    return_variables = []
            else:
                return_variables = self.return_variables
                add_prefix = True
        if return_variables is not None:
            self._return_variables(
                instance = instance,
                variables = return_variables,
                add_prefix = add_prefix)
        return self

    def _return_variables(self, instance, variables, add_prefix = False):
        if not self.exists('returned_variables'):
            self.returned_variables = []
        for variable in self.listify(variables):
            if hasattr(instance, variable):
                if add_prefix:
                    name = instance.name + '_' + variable
                else:
                    name = variable
                setattr(self, name, getattr(instance, variable))
                self.returned_variables.append(name)
            elif self.verbose:
                print(variable, 'not found in', instance.name)
        return self

    """ Core siMpLify methods """

    def draft(self):
        """Creates initial settings for class based upon Idea settings."""
        self.checks.append('package')
        super().draft()
        return self

    def edit_steps(self, steps):
        """Adds a step or list of steps to the 'steps' attribute.

        Args:
            steps(SimplePackage, SimpleTechnique, list(SimplePackage),
                  list(SimpleTechnique)): step(s) to be added into 'steps'
                  attribute.
        
        Raises:
            KeyError if step in steps is not in options. To avoid this error,
                either pass steps as a dict or edit the options first (using
                'edit' method).
        """
        if isinstance(steps, dict):
            self.steps.update(steps)
        elif isinstance(steps, list):
            for step in steps:
                if not step in self.options:
                    error = 'cannot add step that is not in options'
                    raise KeyError(error)
                self.steps.update({step: self.options[step]})
                if not step in self.order:
                    self.order.append(step)        
        return self


@dataclass
class SimpleComparer(SimplePackage):
    """
    """
    name: str = 'generic_comparer'
    options: object = None
    steps: object = None
    plans: object = None
    auto_draft: bool = True
    auto_draft: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Dunder Methods """
    
    def __iter__(self):
        """Allows class instance to be directly iterated.
        
        Returns:
            dict.items(): the iterable 'items' attribute of the dict attribute
                stored in the value of 'comparer_iterable'.
        """
        return getattr(self, self.comparer_iterable).items()
        
    """ Private Methods """
    
    def _check_comparer(self):
        """Checks if necessary attributes exist for comparisons.
        
        If such attributes do not exist, they are created with default values.        
        """
        if not self.exists('comparer_iterable'):
            self.comparer_iterable = 'plans'  
        if getattr(self, self.comparer_iterable) is None:
            setattr(self, self.comparer_iterable, {})        
        return self

    def _draft_comparer(self):
        """Injects Comparer class with needed attributes."""
        setattr(self.comparer, 'options', self.options)
        setattr(self.comparer, 'order', self.order)        
        return self
    
    """ Core siMpLify methods """

    def draft(self):
        """Creates initial settings for class based upon Idea settings."""
        self.checks.append('comparer')
        super().draft()
        return self

    def edit_comparer(self, comparers):
        """Adds a comparer or list of comparers to the attribute named in
        'comparer_iterable'.

        Args:
            comparers(dict(str/int: SimplePlan or list(dict(str/int: 
                SimplePlan)): plan(s) to be added to the attribute named in
                'comparer_iterable'.
        """
        if isinstance(comparers, dict):
            comparers = list(comparers.values())
            last_num = list(self.comparers.keys())[-1:]
        else:
            last_num = 0
        for i, comparer in enumerate(self.listify(comparers)):
            self.comparers.update({last_num + i + 1: comparer})
        return self

    def publish(self):
        super().publish()

        for i, plan in enumerate(self.plans):
            steps = {}
            for j, technique in enumerate(plan):
                steps.update({self.order[j]: technique})
            getattr(self, self.comparer_iterable).update(
                    {str(i + 1): self.comparer(number = i + 1, steps = steps)})
        for step in self.order:
            setattr(self, step, self.options[step](
                technique = self.steps[step]))
        return self

    def implement(self, *args, **kwargs):
        """Method that implements all of the publishd objects with the passed
        args and kwargs.

        If 'return_variables' is defined. Those named attributes are copied from
        step class instances to the present class.

        Args:
            *args, **kwargs: other parameters can be added to method as needed.

        """
        for step in self.order:
            getattr(self, step).implement(*args, **kwargs)
            if self.exists('return_variables'):
                self._infuse_return_variables(instance = getattr(self, step))
        return self


@dataclass
def SimpleSequencer(SimplePackage):
    """
    """
    name: str = 'generic_sequencer'
    options: object = None
    steps: object = None
    auto_draft: bool = True
    auto_draft: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Dunder Methods """
    
    def __iter__(self):
        """Allows class instance to be directly iterated.
        
        Returns:
            dict.items(): the iterable 'items' attribute of the 'steps' dict.
        """
        return self.steps.items()
        
    """ Core siMpLify methods """

    def publish(self):
        super().publish()
        for step in self.order:
            self.steps.update(step, self.options[step]())
        return self

    def implement(self, *args, **kwargs):
        """Method that implements all of the publishd objects with the passed
        args and kwargs.

        If 'return_variables' is defined. Those named attributes are copied from
        step class instances to the present class.

        Args:
            *args, **kwargs: other parameters can be added to method as needed.

        """
        for step, technique in self.steps.items():
            technique.implement(*args, **kwargs)
            if self.exists('return_variables'):
                self._infuse_return_variables(instance = technique)
        return self