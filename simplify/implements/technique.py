
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

    def prepare(self):
        self.tool = self.techniques[self.technique](**self.parameters)
        return self

    def start(self, ingredients):
        ingredients = self.tool(ingredients)
        return ingredients