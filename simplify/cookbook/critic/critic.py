
from dataclasses import dataclass

from .presentation import Presentation
from .review import Review
from ...core.base import SimpleClass


@dataclass
class Critic(SimpleClass):

    menu : object
    inventory : object
    name : str = 'critic'
    auto_prepare : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def _inject(self):
        self.menu.inject(instance = Review,
                         sections = ['general', 'cookbook', 'files', 'review'])
        self.menu.inject(instance = Presentation,
                         sections = ['general', 'files', 'review',
                                     'presentation'])
        return self

    def _set_defaults(self):
        self._inject()
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