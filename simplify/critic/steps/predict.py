
from dataclasses import dataclass


from simplify.core.base import SimpleStep


@dataclass
class Predict(SimpleStep):
    """Core class for making predictions based upon machine learning models.

    """
    techniques : object = None
    name : str = 'predictor'
    auto_finalize : bool = True
    auto_produce : bool = False

    def __post_init__(self):
        """Sets up the core attributes of an Evaluator instance."""
        super().__post_init__()
        return self


    def _predict_outcomes(self, recipe):
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

    def _predict_probabilities(self, recipe):
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

    def draft(self):
        super().draft()
        self.options = {'outcomes' : self._predict_outcomes,
                        'probabilities' : self._predict_probabilities}
        return self

    def produce(self, recipe):
        self.predictions = self.options[self.technique](recipe = recipe)
        return self

