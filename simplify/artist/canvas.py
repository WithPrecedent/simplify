"""
.. module:: canvas
:synopsis: data visualizations
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.chapter import Chapter
from simplify.core.book import Book


@dataclass
class Canvas(Book):
    """Builds tools for data visualization.

    Args:
        ingredients(Ingredients): an instance of Ingredients. This argument need
            not be passed when the class is instanced. It can be passed directly
            to the 'publish' method as well.
        steps(dict(str: ArtistTechnique)): names and related ArtistTechnique
            classes for analyzing fitted models.
        recipes(Recipe or list(Recipe)): a list or single Recipe to be reviewed.
            This argument need not be passed when the class is instanced. It
            can be passed directly to the 'publish' method as well.
        reviews(Review): an instance of Review containing all metrics and
            evaluation results.This argument need not be passed when the class
            is instanced. It can be passed directly to the 'publish' method as
            well.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_draft (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.

    Since this class is a subclass to SimpleIterable and SimpleContributor, all
    documentation for those classes applies as well.

    """

    ingredients: object = None
    steps: object = None
    recipes: object = None
    reviews: object = None
    name: str = 'canvas'
    auto_draft: bool = True
    auto_publish: bool = True

    def __post_init__(self) -> None:
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

    def draft(self) -> None:
        """Sets default styles, options, and plots."""

        self._set_styler()
        return self

    def publish(self, data = None, recipes = None, reviews = None):
        if self.ingredients is None:
            self.data = self.recipes.ingredients
        for name, step  in self.steps:
            for recipe in listify(self.recipes):
                if self.verbose:
                    print('Visualizing', recipe.name + recipe.number)
                for step, step in getattr(self, self.iterator).items():
                    step.implement(recipe = recipe, review = reviews)
        return self


@dataclass
class Illustration(Chapter):

    def __post_init__(self) -> None:
        super().__post_init__()
        return self