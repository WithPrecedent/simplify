
"""
.. module:: technique
:synopsis: technique in siMpLify step
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.step import SimpleStep


@dataclass
class SimpleTechnique(SimpleStep):
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
        """Finalizes parameters and adds 'parameters' to 'algorithm'."""
        self._publish_parameters()
        if self.technique != ['none']:
            self.algorithm = self.options[self.technique](**self.parameters)
        else:
            self.algorithm = None
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
