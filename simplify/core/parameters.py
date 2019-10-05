"""
.. module:: parameters
:synopsis: aggregates, selects, and finalizes parameters
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.base import SimpleClass


@dataclass
class SimpleParameters(SimpleClass):

    auto_publish : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Magic Methods """

    def __call__(self, instance):
        self.__post_init__()
        return self.produce(instance = instance)

    """ Private Methods """

    def _denestify(self, outer_key, parameters):
        """Removes outer layer of 'parameters' dict, if it exists, by using
        'outer_key' as the key.

        Args:
            outer_key(str): name of key to use if 'parameters' is nested.
            parameters(dict): one- or two-level dictionary of parameters.

        Returns:
            parameters(dict): if passed 'parameters' are nested, the nested
                dictionary at 'outer_key' is returned. If 'parameters' is not
                nested, 'parameters' is returned unaltered.

        """
        if self.is_nested(parameters) and outer_key in parameters:
            return parameters[outer_key]
        else:
            return parameters

    def _get_parameters(self, instance, parameters):
        """Returns initial or default parameters to be processed from instance.

        Args:
            instance(object): subclass for parameters to be added.

        Returns
            parameters(dict): an initialized dictionary of parameters.

        """
        if not (instance.exists('parameters') or instance.parameters):
            if instance.exists('default_parameters'):
                if instance.exists('technique'):
                    parameters = self._denestify(
                        outer_key = instance.technique,
                        parameters = instance.default_parameters)
                else:
                    parameters = instance.default_parameters
        else:
            if instance.exists('technique'):
                parameters = self._denestify(
                    outer_key = instance.technique,
                    parameters = instance.parameters)
            else:
                parameters = instance.parameters
        return parameters

    def _get_parameters_selected(self, instance, parameters):
        """For subclasses that only need a subset of the parameters stored in
        idea, this function selects that subset.

        Args:
            parameters_to_use(list or str): list or string containing names of
                parameters to include in final parameters dict.
        """
        if (instance.exists('selected_parameters')
                and instance.selected_parameters
                and instance.exists('default_parameters')):
            if instance.exists('technique'):
                parameters_to_use = list(self._denestify(
                        instance.technique,
                        instance.default_parameters).keys())
            else:
                parameters_to_use = instance.default_parameters.keys()
            new_parameters = {}
            for key, value in parameters.items():
                if key in self.listify(parameters_to_use):
                    new_parameters.update({key: value})
            parameters = new_parameters
        return parameters

    def _is_nested(self, parameters):
        """Returns if passed 'parameters' is nested at least one-level.

        Args:
            parameters(dict): dict to be tested.

        Returns:
            boolean value indicating whether any value in the 'parameters' is
                also a dict (meaning that 'parameters' is nested).
        """
        return any(isinstance(d, dict) for d in parameters.values())

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {'parameters': 'parameters',
                        'selected': 'selected_parameters',
                        'runtime': 'runtime_parameters',
                        'extra': 'extra_parameters',
                        'conditional': 'conditional_parameters'}
        return self

    def publish(self):
        pass
        return self

    def implement(self, instance, parameter_types = None):
        """Returns parameters from different possible sources based upon
        instance attributes.

        Args:
            instance(object): subclass for parameters to be added.
            parameter_types(list(str) or str): attribute names of parameter
                groups in the instance class. If not provided, all of the keys
                in 'options' will be incorporated if they exist.

        Returns
            parameters(dict): a finalized dictionary of parameters.

        """
        parameters = {}
        # Sets which groupings of parameters to use.
        if parameter_types:
            parameter_types = self.listify(parameter_types)
        else:
            parameter_types = list(self.options.keys())
        # Iterates through possible parameter groups and adjusts 'parameters'
        for key, value in self.options.items():
            if key in parameter_types:
                if ('key' in ['conditional']
                        and hasattr(instance, '_get_parameters_conditional')):
                    parameters = instance._get_parameters_conditional(
                            parameters = parameters)
                else:
                    if hasattr(self, '_get_' + value):
                        parameters = getattr('_get_' + value)(
                                instance = instance, parameters = parameters)
                    elif instance.exists(value):
                        if instance.exists('technique'):
                            parameters.update(self._denestify(
                                    outer_key = instance.technique,
                                    parameters = getattr(instance, value)))
                        else:
                            parameters.update(getattr(instance, value))
        return parameters
