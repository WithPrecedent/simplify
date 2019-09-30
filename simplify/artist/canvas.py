"""
canvas.py is the primary control file for data visualization in the siMpLify
package.

Contents:

    Canvas: class which handles construction and utilization of visuals
        to create charts, graphs, and other visuals in the siMpLify package.
    Illustration: class which stores a particular set of techniques and
        algorithms used to create data visualizations.

    Both classes are subclasses to SimpleClass and follow its structural rules.

"""
from dataclasses import dataclass

from simplify.artist.steps.style import Style
from simplify.artist.steps.paint import Paint
from simplify.artist.steps.animate import Animate
from simplify.core.base import SimpleClass, SimplePlan


@dataclass
class Canvas(SimpleClass):
    """Visualizes data and analysis based upon the nature of the machine
    learning model used in the siMpLify package.
    """
    ingredients : object = None
    steps : object = None
    recipes : object = None
    name : str = 'canvas'
    planner_type : str = 'serial'
    auto_finalize : bool = True
    auto_produce : bool = True

    def __post_init__(self):
        self.styler = []
        super().__post_init__()
        return self

    """ Private Methods """

    def _check_model_type(self):
        """Sets default paintings, animations, and any other added options if
        user selects 'default' as the option for the respective option.
        """
        for key in self.options.keys():
            if not hasattr(self, key) and getattr(self, key) == 'default':
                setattr(self, key, getattr(
                    self, '_default_' + self.model_type + 'key')())
        return self

    def _default_classifier_animations(self):
        """Returns list of default animations for classifier algorithms."""
        return []

    def _default_classifier_paintings(self):
        """Returns list of default plots for classifier algorithms."""
        return ['confusion', 'heat_map', 'ks_statistic', 'pr_curve',
                'roc_curve']

    def _default_cluster_animations(self):
        """Returns list of default animations for cluster algorithms."""
        return []

    def _default_cluster_paintings(self):
        """Returns list of default plots for cluster algorithms."""
        return ['cluster_tree', 'elbow', 'silhouette']

    def _default_regressor_animations(self):
        """Returns list of default animations for regressor algorithms."""
        return []

    def _default_regressor_paintings(self):
        """Returns list of default plots for regressor algorithms."""
        return ['heat_map', 'linear', 'residuals']

    def _get_ingredients(self, recipe = None):
        """
        """
        if recipe:
            return recipe.ingredients
        elif self.ingredients is not None:
            return self.ingredients
        else:
            error = 'produce method requires Ingredients or Recipe instance'
            raise TypeError(error)

    def _set_styler(self):
        if 'styler' not in self.steps:
            self.steps = ['styler'] + self.steps
        return self

    """ Core Public siMpLify Methods """

    def draft(self):
        """Sets default styles, options, and plots."""
        self.options = {
                'styler' : ['simplify.artist.illustrate', 'Style'],
                'illustrate': ['simplify.artist.illustrate', 'Illustrate'],
                'paint': ['simplify.artist.paint', 'Paint'],
                'animate': ['simplify.artist.animate', 'Animate']}
        self.checks = ['steps', 'model_type']
        # Locks 'step' attribute at 'artist' for conform methods in package.
        self.step = 'artist'
        # Sets 'manager_type' so that proper parent methods are used.
        self.manager_type = 'serial'
        # Sets 'plan_class' to allow use of parent methods.
        self.plan_class = Illustration
        self.plan_iterable = 'illustrations'

        return self

    def finalize(self):
        self._set_styler()
        super().finalize()
        self.illustrations = Illustration()
        return self

    def produce(self, recipes = None, reviews = None):
        for recipe in self.listify(recipes):
            if self.verbose:
                print('Evaluating', recipe.name + 's')
            for step, technique in getattr(self, self.plan_iterable).items():
                technique.produce(recipe = recipe, review = review)
        return self


@dataclass
class Illustration(SimplePlan):

    def __post_init__(self):
        super().__post_init__()
        return self

    def produce(self, recipes, reviews):
        for i, recipe in enumerate(self.listify(recipes)):
            review = self.listify(reviews)[i]
            for step, technique in self.techniques.items():
                recipe = technique.produce(recipe = recipe, review = review)
        return self