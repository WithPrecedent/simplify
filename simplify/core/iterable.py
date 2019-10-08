"""
.. module:: iterables
:synopsis: iterable builder and container
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.base import SimpleClass


@dataclass
class SimpleIterable(SimpleClass):
    """Parent class for building and storing iterable steps.

    This class adds methods useful to create iterators, iterate over user
    options, and transform data or fit models. SimpleIterable classes define
    the bulk of the siMpLify processing packages (Farmer, Chef, Critic, Artist)
    with only the lowest layer of the class hierarchy in a project being
    SimpleTechnique subclasses.

    To take maximum advantage of this class's functionality, a subclass should
    either, in its draft method, call super().draft() and/or define the
    following attributes there:
        iterable_setting(str): name of key in an Idea instance where the
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
    number: int = 0
    name: str = 'simple_iterable'

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Magic Methods """

    def __call__(self, *args, **kwargs):
        """Calls '__post_init__ and then 'implement' with args and kwargs."""
        self.__post_init__()
        return self.implement(*args, **kwargs)

    def __iter__(self):
        """Allows class instance to be directly iterated by returning the
        primary iterable contained within the class instance.
        """
        return getattr(self, self.iterator)

    """ Private Methods """

    def _add_list_to_iterator(self, items):
        for step in self.listify(items):
            getattr(self, self.iterator).update({step: self.options[step]})
        return self
            
    def _check_iterator(self):
        """Creates class iterable attribute to be filled with concrete steps if
        one does not exist.
        """
        if not self.exists('iterator'):
            self.iterator = 'steps'
        if not self.exists(self.iterator):
            setattr(self, self.iterator, {})
        if not self.exists('iterable_setting'):
            self.iterable_setting = self.name + '_steps'
        if self.exists(self.iterable_setting):
            self._add_list_to_iterator(
                self._convert_wildcards(getattr(self, self.iterable_setting)))
        else:
            setattr(self, self.iterator, self.options)
        return self

    def _infuse_attributes(self, instance, return_variables = None):
        """Adds 'return_variables' attributes from instance class to present
        class.

        Args:
            instance(object): class instance with attributes to be added to the
                present subclass instance.
            return_variables(list(str) or dict(str: list(str))): names of
                attributes sought. If stored in a dict, the outer key
                corresponds to particular keys stored in 'options'.

        """
        if return_variables is None and self.exists('return_variables'):
            if isinstance(self.return_variables, dict):
                return_variables = self.return_variables[instance.name]
            else:
                return_variables = self.return_variables
        if return_variables is not None:
            for variable in self.listify(return_variables):
                if hasattr(instance, variable):
                    setattr(self, variable, getattr(instance, variable))
                elif self.verbose:
                    print(variable, 'not found in', instance.name)
        return self

    """ Core siMpLify methods """

    def draft(self):
        """ Declares defaults for class."""
        super().draft()
        self.options = {}
        self.checks.extend(['iterator'])
        self.return_variables = []
        self.iterator = 'steps'
        return self

    def edit_iterable(self, iterables):
        """Adds a single iterable or list of iterables to the iterables
        attribute.

        Args:
            iterables(SimpleIterable, list(SimpleIterable)): iterables to be
                added into 'iterables' attribute.
        """
        if isinstance(iterables, dict):
            getattr(self, self.iterator).update(iterables)
        elif isinstance(iterables, list):
            self._add_list_to_iterable(items = iterables)
        return self

    def publish(self):
        super().publish()
        for name, step in getattr(self, self.iterator).items():
            print('name', name)
            print('step', step)
            if isinstance(step, str):
                print('step is string')
                setattr(self, name, self.options[name](technique = step))
            else:
                setattr(self, name, step())
        return self
    
    def implement(self, *args, **kwargs):
        """Method that implements all of the publishd objects with the passed
        args and kwargs.

        If 'return_variables' is defined. Those named attributes are copied from
        step class instances to the present class.

        Args:
            *args, **kwargs: other parameters can be added to method as needed.

        """
        for name in getattr(self, self.iterator).keys():
            getattr(self, name).implement(*args, **kwargs)
            if self.exists('return_variables'):
                self._infuse_attributes(instance = getattr(self, name))
        return self
