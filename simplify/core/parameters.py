"""
.. module:: parameters
:synopsis: aggregates, selects, and finalizes parameters
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.base import SimpleClass


@dataclass
class SimpleParameters(SimpleClass):

    auto_publish : bool = True
    
    def __post_init__(self):
        super().__post_init__()
        return self
        
    """ Magic Methods """
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return self.final_parameters
    
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

    def _get_parameters(self, technique, parameters):
        """Returns parameters from different possible sources based upon passed
        'technique'.

        If 'parameters' attribute is None, the accessible Idea instance is
        checked for a section matching the 'name' attribute of the class
        instance. If no Idea parameters exist, 'default_parameters' are used.
        If there are no 'default_parameters', an empty dictionary is created
        for parameters.

        Args:
            technique(str): name of technique for which parameters are sought.

        """
        if self.exists('parameters') and self.parameters:
            self.parameters = self._denestify(self.technique, self.parameters)
        elif self.technique in self.idea.configuration:
            self.parameters = self.idea.configuration[self.technique]
        elif self.name in self.idea.configuration:
            self.parameters = self.idea.configuration[self.name]
        elif self.exists('default_parameters'):
            self.parameters = self._denestify(
                    technique = self.technique,
                    parameters = self.default_parameters)
        else:
            self.parameters = {}
        return self

    def _get_parameters_extra(self, technique, parameters):
        """Adds parameters from 'extra_parameters' if attribute exists.

        Some parameters are stored in 'extra_parameters' because of the way
        the particular algorithms are constructed by dependency packages. For
        example, scikit-learn consolidates all its support vector machine
        classifiers into a single class (SVC). To pick different kernels for
        that class, a parameter ('kernel') is used. Since siMpLify wants to
        allow users to compare different SVC kernel models (linear, sigmoid,
        etc.), the 'extra_parameters attribute is used to add the 'kernel'
        and 'probability' paramters in the Classifier subclass.

        Args:
            technique (str): name of technique selected.
            parameters (dict): a set of parameters for an algorithm.

        Returns:
            parameters (dict) with 'extra_parameters' added if that attribute
                exists in the subclass and the technique is listed as a key
                in the nested 'extra_parameters' dictionary.
        """
        if self.exists('extra_parameters') and self.extra_parameters:

            self.parameters.update(
                    self._denestify(technique = technique,
                                    parameters = self.extra_parameters))
            return

    def _get_parameters_runtime(self, technique, parameters):
        """Adds runtime parameters to parameters based upon passed 'technique'.

        Args:
            technique(str): name of technique for which runtime parameters are
                sought.
        """
        if self.exists('runtime_parameters'):
            self.parameters.update(
                    self._denestify(technique = technique,
                                    parameters = self.runtime_parameters))
            return parameters


    def _get_parameters_selected(self, technique, parameters,
                                 parameters_to_use = None):
        """For subclasses that only need a subset of the parameters stored in
        idea, this function selects that subset.

        Args:
            parameters_to_use(list or str): list or string containing names of
                parameters to include in final parameters dict.
        """
        if self.exists('selected_parameters') and self.selected_parameters:
            if not parameters_to_use:
                if isinstance(self.selected_parameters, list):
                    parameters_to_use = self.selected_parameters
                elif self.exists('default_parameters'):
                    parameters_to_use = list(self._denestify(
                            technique, self.default_parameters).keys())
            new_parameters = {}
            for key, value in parameters.items():
                if key in self.listify(parameters_to_use):
                    new_parameters.update({key: value})
            self.parameters = new_parameters
        return self 
    
    def _is_nested(self, parameters):
        """Returns if passed 'parameters' is nested at least one-level.

        Args:
            parameters(dict): dict to be tested.

        Returns:
            boolean value indicating whether any value in the 'parameters' is
                also a dict (meaning that 'parameters' is nested).
        """
        return any(isinstance(d, dict) for d in parameters.values())
    
    def _publish_parameters(self):
        """Compiles appropriate parameters for all 'technique'.

        After testing several sources for parameters using '_get_parameters',
        parameters are subselected, if necessary, using '_select_parameters'.
        If 'runtime_parameters' and/or 'extra_parameters' exist in the
        subclass, those are added to 'parameters' as well.
        """
        parameter_groups = ['', '_selected', '_runtime', '_extra',
                            '_conditional']
        for parameter_group in parameter_groups:
            if hasattr(self, '_get_parameters' + parameter_group):
                getattr(self, '_get_parameters' + parameter_group)(
                        technique = self.technique,
                        parameters = self.parameters)

        return self
   
    """ Public Tool Methods """
    
    def inject(self, instance, parameter_types = None, override = False):
        """Stores the section or sections of the 'configuration' dictionary in
        the passed class instance as attributes to that class instance. 
        
        If the sought section has the '_parameters' suffix, the section is 
        returned as a single dictionary at instance.parameters (assuming that 
        it does not exist or 'override' is True).
        
        If the sought key from a section has the '_steps' suffix, the value for
        that key is stored at instance.steps (assuming that it does not exist or
        'override' is True).
        
        If the sought key from a section has the '_techniques' suffix, the value 
        for that key is stored either at the attribute named the prefix of the 
        key (assuming that it does not exist or 'override' is True).
        
        Wildcard values of 'all', 'default', and 'none' are appropriately 
        changed with the '_convert_wildcards' method.

        Args:
            instance(object): a class instance to which attributes should be 
                added.
            override(bool): if True, even existing attributes in instance will
                be replaced by configuration parameter items. If False,
                current values in those similarly-named parameters will be
                maintained (unless they are None).

        Returns:
            instance with attribute(s) added.
            
        """
        if parameter_types:
            self.parameter_types = self.listify(parameter_types)
        else:
            self.parameter_types = list(self.options.keys())
        for key, value in self.options.items():
            if (key.endswith('_steps') 
                    and (not instance.exists('steps') or override)):
                instance.steps = instance._convert_wildcards(value)
            elif key.endswith('_technique'):
                attribute_name = key.replace('_technique', '')
                if not instance.exists(attribute_name) or override:
                    setattr(instance, attribute_name, 
                            instance._convert_wildcards(value))    
            elif not instance.exists(key) or override:
                setattr(instance, key, 
                        instance._convert_wildcards(value))
        return instance
     
    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {'parameters': 'parameters',
                        'selected': 'selected_parameters',
                        'runtime': 'runtime_parameters',
                        'extra': 'extra_parameters',
                        'conditional': 'conditional_parameters'}
        return self

    def edit(self, technique, parameters):
        """Adds a parameter set to parameters dictionary.

        Args:
            parameters(dict): dictionary of parameters to be added to
                'parameters' of subclass.

        Raises:
            TypeError: if 'parameters' is not dict type.
        """
        if isinstance(parameters, dict):
            if not hasattr(self, 'parameters') or self.parameters is None:
                self.parameters = {technique: parameters}
            else:
                self.parameters[technique].update(parameters)
            return self
        else:
            error = 'parameters must be a dict type'
            raise TypeError(error)
        
    def publish(self):
        self.technique = self._convert_wildcards(value = self.technique)
        self._publish_parameters()
        if self.technique in ['none', 'None', None]:
            self.technique = 'none'
            self.algorithm = None
        elif (self.exists('custom_options')
                and self.technique in self.custom_options):
            self.algorithm = self.options[self.technique](
                    parameters = self.parameters)
        return self