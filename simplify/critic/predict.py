"""
.. module:: predict
:synopsis: creates predictions from machine learning models
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass


from simplify.core.base import SimplePlan, SimpleStep


@dataclass
class Predict(SimplePlan):
    """Creates predictions from fitted models for out-of-sample data.
    
    Args:
        steps(dict(str: SimpleStep)): names and related SimpleStep classes for
            explaining data analysis models.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_produce (bool): whether to call the 'produce' method when the class
            is instanced.
    """
    steps: object = None
    name: str = 'predictor'
    auto_publish: bool = True
    auto_produce: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
                'outcomes': PredictOutcomes,
                'probabilities': PredictProbabilities,
                'log_probabilities': PredictLogProbabilities,
                'shap': PredictShapProbabilities}
        return self

    def produce(self, recipe):
        for step_name, step_instance in self.steps.items():
            setattr(self, step_name, step_instance.produce(recipe = recipe))
        return self


@dataclass
class PredictOutcomes(SimpleStep):
    """Estimates outcomes based upon fit model.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'outcome_predictor'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def publish(self):
        pass
        return self

    def produce(self, recipe):
        """Makes predictions from fitted model.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predictions from fitted model on test data.
        """
        if hasattr(self.recipe.model.algorithm, 'predict'):
            return self.recipe.model.algorithm.predict(
                self.recipe.ingredients.x_test)
        else:
            if self.verbose:
                print('predict method does not exist for',
                    self.recipe.model.technique.name)
            return None


@dataclass
class PredictProbabilities(SimpleStep):
    """Estimates probabilities of outcomes based upon fit model.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'probabilities_predictor'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def publish(self):
        pass
        return self

    def produce(self, recipe):
        """Makes predictions from fitted model.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predicted probabilities from fitted model on test data.
        """
        if hasattr(self.recipe.model.algorithm, 'predict_proba'):
            return self.recipe.model.algorithm.predict_proba(
                    self.recipe.ingredients.x_test)
        else:
            if self.verbose:
                print('predict_proba method does not exist for',
                    self.recipe.model.technique.name)
            return None


@dataclass
class PredictLogProbabilities(SimpleStep):
    """Estimates log probabilities of outcomes based upon fit model.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'log_probabilities_predictor'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def publish(self):
        pass
        return self

    def produce(self, recipe):
        """Makes predictions from fitted model.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predicted probabilities from fitted model on test data.
        """
        if hasattr(self.recipe.model.algorithm, 'predict_log_proba'):
            return self.recipe.model.algorithm.predict_log_proba(
                    self.recipe.ingredients.x_test)
        else:
            if self.verbose:
                print('predict_log_proba method does not exist for',
                    self.recipe.model.technique.name)
            return None

@dataclass
class PredictShapProbabilities(SimpleStep):
    """Estimates probabilities of outcomes based upon fit model using SHAP
    values.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'log_probabilities_predictor'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def publish(self):
        pass
        return self

    def produce(self, recipe):
        """Makes predictions from fitted model.

        Args:
            recipe(Recipe): instance of Recipe with a fitted model.

        Returns:
            Series with predicted probabilities from fitted model on test data.
        """
        return None