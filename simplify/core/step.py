"""
.. module:: step
:synopsis: iterable step for siMpLify plan
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.base import SimpleClass


@dataclass
class SimpleStep(SimpleClass):
    """Parent class for various steps in the siMpLify package.

    SimpleStep, unlike the above subclasses of SimpleClass, should have a
    'parameters' parameter as an attribute to the class instance for the
    included methods to work properly. Otherwise, 'parameters' will be set to
    an empty dict.

    'fit', 'fit_transform', and 'transform' adapter methods are included in
    SimpleClass to support partial scikit-learn compatibility.

    Args:
        techniques(list of str): name of technique(s) that match(es) string(s)
            in the 'options' keys or a wildcard value such as 'default', 'all',
            or 'none'.
        parameters(dict): parameters to be attached to algorithm in 'options'
            corresponding to 'techniques'. This parameter need not be passed to
            the SimpleStep subclass if the parameters are in the accessible
            Idea instance or if the user wishes to use default parameters.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.
    """

    technique: object = None
    parameters: object = None
    auto_publish: bool = True

    def __post_init__(self):
        # Adds name of SimpleManager subclass to sections to inject from Idea
        # so that all of those section entries are available as local
        # attributes.
        if self.exists('manager_name'):
            self.idea_sections = [self.manager_name]
        super().__post_init__()
        return self

    """ Private Methods """

    def _check_parameters(self):
        """Adds empty 'parameters' dict if it doesn't exist."""
        if not self.exists('parameters'):
            self.parameters = {}
        return self


    def _denestify(self, technique, parameters):
        """Removes outer layer of 'parameters' dict, if it exists, by using
        'technique' as the key.

        If 'parameters' is not nested, 'parameters' is returned unaltered.
        """
        if self.is_nested(parameters) and technique in parameters:
            return parameters[technique]
        else:
            return parameters

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
            parameters.update(
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
            parameters.update(
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

    """ Core siMpLify Public Methods """

    def draft(self):
        """Default draft method which sets bare minimum requirements.

        This default draft should only be used if users are planning to
        manually add all options and parameters to the SimpleStep subclass.
        """
        self.options = {}
        self.checks = ['idea', 'parameters']
        return self

    def edit_parameters(self, technique, parameters):
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
        """Finalizes parameters and adds 'parameters' to 'algorithm'."""
        self.technique = self._convert_wildcards(value = self.technique)
        self._publish_parameters()
        if self.technique in ['none', 'None', None]:
            self.technique = 'none'
            self.algorithm = None
        elif (self.exists('custom_options')
                and self.technique in self.custom_options):
            self.algorithm = self.options[self.technique](
                    parameters = self.parameters)
        else:
            self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def read(self, ingredients, plan = None):
        """Generic implementation method for SimpleStep subclass.

        This method should only be used if the algorithm is to be applied to
        'x' and 'y' in ingredients and sklearn compatible 'fit' and 'transform'
        methods are available.

        Args:
            ingredients (Ingredients): an instance of Ingredients or subclass.
            plan (SimplePlan subclass or instance): is not used by the generic
                method but is made available as an optional keyword for
                compatibility with other 'read'  methods. This parameter is
                used when the current SimpleStep subclass needs to look back at
                previous SimpleSteps (as in Cookbook steps).
        """
        if self.algorithm != 'none':
            self.algorithm.fit(ingredients.x_train, ingredients.y_train)
            ingredients.x_train = self.algorithm.transform(ingredients.x_train)
        return ingredients


    """ Scikit-Learn Compatibility Methods """

    def fit(self, x = None, y = None, ingredients = None):
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x(DataFrame or ndarray): independent variables/features.
            y(DataFrame, Series, or ndarray): dependent variable(s)/feature(s)
            ingredients(Ingredients): instance of Ingredients containing
                x_train and y_train attributes (based upon possible remapping).

        Raises:
            AttributeError if no 'fit' method exists for local 'algorithm'.
        """
        if hasattr(self.algorithm, 'fit'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    self.algorithm.fit(x)
                else:
                    self.algorithm.fit(x, y)
            elif ingredients is not None:
                ingredients = self.algorithm.fit(ingredients.x_train,
                                                 ingredients.y_train)
        else:
            error = 'fit method does not exist for this algorithm'
            raise AttributeError(error)
        return self

    def fit_transform(self, x = None, y = None, ingredients = None):
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x(DataFrame or ndarray): independent variables/features.
            y(DataFrame, Series, or ndarray): dependent variable(s)/feature(s)
            ingredients(Ingredients): instance of Ingredients containing
                x_train and y_train attributes (based upon possible remapping).

        Returns:
            transformed x or ingredients, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or ingredients is not passed to
                the method.
        """
        self.fit(x = x, y = y, ingredients = ingredients)
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            return self.transform(x = x, y = y)
        elif ingredients is not None:
            return self.transform(ingredients = ingredients)
        else:
            error = 'fit_transform requires DataFrame, ndarray, or Ingredients'
            raise TypeError(error)

    def transform(self, x = None, y = None, ingredients = None):
        """Generic transform method for partial compatibility to sklearn.
        Args:
            x(DataFrame or ndarray): independent variables/features.
            y(DataFrame, Series, or ndarray): dependent variable(s)/feature(s)
            ingredients(Ingredients): instance of Ingredients containing
                x_train and y_train attributes (based upon possible remapping).

        Returns:
            transformed x or ingredients, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'algorithm'.
        """
        if hasattr(self.algorithm, 'transform'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    x = self.algorithm.transform(x)
                else:
                    x = self.algorithm.transform(x, y)
                return x
            elif ingredients is not None:
                ingredients = self.algorithm.transform(ingredients.x_train,
                                                       ingredients.y_train)
                return ingredients
        else:
            error = 'transform method does not exist for this algorithm'
            raise AttributeError(error)
