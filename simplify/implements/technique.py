
from dataclasses import dataclass
import re

import pandas as pd


@dataclass
class Technique(object):

    def __post_init__(self):
        if hasattr(self, '_set_defaults'):
            getattr(self, '_set_defaults')()
        if hasattr(self, '_set_folders'):
            getattr(self, '_set_folders')()
        self.prepare()
        return self

    def __call__(self, *args, **kwargs):
        return self.start(*args, **kwargs)

    def __str__(self):
        """Returns lowercase name of class."""
        return self.__class__.__name__.lower()

    def prepare(self):
        self.tool = self.options[self.technique](**self.parameters)
        return self

    def start(self, ingredients):
        ingredients = self.tool(ingredients)
        return ingredients