"""
.. module:: canvas
:synopsis: data visualizations
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.manager import SimpleManager
from simplify.core.decorators import localize


@dataclass
class Canvas(SimpleManager):
    """Builds tools for data visualization.

    Args:
        ingredients(Ingredients): an instance of Ingredients. This argument need
            not be passed when the class is instanced. It can be passed directly
            to the 'implement' method as well.
        steps(dict(str: SimpleStep)): names and related SimpleStep classes for
            analyzing fitted models.
        recipes(Recipe or list(Recipe)): a list or single Recipe to be reviewed.
            This argument need not be passed when the class is instanced. It
            can be passed directly to the 'implement' method as well.
        reviews(Review): an instance of Review containing all metrics and
            evaluation results.This argument need not be passed when the class
            is instanced. It can be passed directly to the 'implement' method as
            well.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_implement (bool): whether to call the 'implement' method when the class
            is instanced.

    Since this class is a subclass to SimpleManager and SimpleClass, all
    documentation for those classes applies as well.

    """

    ingredients: object = None
    steps: object = None
    recipes: object = None
    reviews: object = None
    name: str = 'canvas'
    auto_publish: bool = True
    auto_implement: bool = True

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
            error = 'implement method requires Ingredients or Recipe instance'
            raise TypeError(error)

    def _set_styler(self):
        if 'styler' not in self.steps:
            self.steps = ['styler'] + self.steps
        return self

    """ Core siMpLify Methods """

    def draft(self):
        """Sets default styles, options, and plots."""
        self.options = {
                'style': ['simplify.artist.syle', 'Style'],
                'illustrate': ['simplify.artist.illustrate', 'Illustrate'],
                'paint': ['simplify.artist.paint', 'Paint'],
                'animate': ['simplify.artist.animate', 'Animate']}
        self.checks = ['steps', 'model_type']
        # Locks 'step' attribute at 'artist' for conform methods in package.
        self.step = 'artist'
        # Sets 'iterable_type' so that proper parent methods are used.
        self.iterable_type = 'serial'
        # Sets 'iterable_class' to allow use of parent methods.
        self.iterable = 'depictions'
        return self

    def publish(self):
        self._set_styler()
        super().publish()
        return self

    @localize
    def implement(self, ingredients = None, recipes = None, reviews = None):
        if self.ingredients is None:
            self.ingredients = self.recipes.ingredients
        for name, step  in self.steps:
            for recipe in self.listify(self.recipes):
                if self.verbose:
                    print('Visualizing', recipe.name + recipe.number)
                for step, technique in getattr(self, self.iterable).items():
                    technique.implement(recipe = recipe, review = reviews)
        return self