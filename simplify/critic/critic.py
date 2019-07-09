
from dataclasses import dataclass

from .presentation import Presentation
from .review import Review

@dataclass
class Critic(object):

    menu : object
    inventory : object
    name : str = 'critic'
    auto_prepare : bool = True

    def __post_init__(self):
        self._set_defaults()
        if self.auto_prepare:
            self.prepare()
        return self

    def _localize(self):
        self.menu.localize(instance = Review,
                           sections = ['general', 'cookbook', 'files',
                                       'review'])
        self.menu.localize(instance = Presentation,
                           sections = ['general', 'files', 'review',
                                       'presentation'])
        return self

    def _set_defaults(self):
        self._localize()
        self.review = Review()
        self.presentation = Presentation(inventory = self.inventory)
        return self

    def prepare(self):
        self.review.prepare()
        self.presentation.prepare()
        return self

    def start(self, recipe):
        self.review.start(recipe = recipe)
        self.presentation.start(recipe = recipe, review = self.review)
        return self