
"""
.. module:: technique
:synopsis: technique in siMpLify step
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.base import SimpleClass
from simplify.core.parameters import SimpleParameters


@dataclass
class SimpleTechnique(SimpleClass):
    """Parent class for various techniques in the siMpLify package.

    SimpleTechnique is the lowest-level parent class in the siMpLify package.
    It follows the general structure of SimpleClass, but is focused on storing
    and applying single techniques to data or other variables. It is included,
    in part, to achieve the highest level of compatibility with scikit-learn as
    currently possible.

    Not every low-level technique needs to a subclass of SimpleTechnique. For
    example, many of the algorithms used in the Cookbook steps (RandomForest,
    XGBClassifier, etc.) are dependencies that are fully integrated into the
    siMpLify architecture without wrapping them into a SimpleTechnique
    subclass. SimpleTechnique is used for custom techniques and for
    dependencies that require a substantial adapter to integrate into siMpLify.

    SimpleTechnique, similar to SimpleStep, should have a 'parameters'
    parameter as an attribute to the class instance for the included methods to
    work properly. Otherwise, 'parameters' will be set to an empty dict.

    Unlike SimpleManager, SimplePlan, and SimpleStep, SimpleTechnique only
    supports a single 'technique'. This is to maximize compatibility to scikit-
    learn and other pipeline scripts.

    Args:
        parameters (dict): parameters to be attached to algorithm in 'options'
            corresponding to 'technique'. This parameter need not be passed to
            the SimpleStep subclass if the parameters are in the accessible
            Idea instance or if the user wishes to use default parameters.
        auto_publish (bool): whether 'publish' method should be called when
            the  class is instanced. This should generally be set to True.

    It is also a child class of SimpleStep. So, its documentation applies as
    well.
    """
    technique: object = None
    parameters: object = None
    auto_publish: bool = True

    def __post_init__(self):
        # Adds name of SimpleStep subclass to sections to inject from Idea
        # so that all of those section entries are available as local
        # attributes.
        if self.exists('step_name'):
            self.idea_sections = [self.step_name]
        super().__post_init__()
        return self

    """ Core siMpLify Public Methods """

    def publish(self):
        self.parameters = SimpleParameters().produce(instance = self)
        if self.exists('technique'):
            if self.technique in ['none', 'None', None]:
                self.technique = 'none'
                self.algorithm = None
            elif (self.exists('simplify_options')
                    and self.technique in self.simplify_options):
                self.algorithm = self.options[self.technique](
                        parameters = self.parameters)
            else:
                self.algorithm = self.options[self.technique](
                        **self.parameters)
        return self

    def implement(self, ingredients, plan = None):
        """Generic implementation method for SimpleTechnique subclass.

        Args:
            ingredients(Ingredients): an instance of Ingredients or subclass.
            plan(SimplePlan subclass or instance): is not used by the generic
                method but is made available as an optional keyword for
                compatibility with other 'implement'  methods. This parameter is
                used when the current SimpleTechnique subclass needs to look
                back at previous SimpleSteps.
        """
        if self.algorithm:
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
