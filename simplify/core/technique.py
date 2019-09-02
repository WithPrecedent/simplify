
from dataclasses import dataclass
import re

import pandas as pd

from .base import SimpleClass

@dataclass
class Technique(SimpleClass):

    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        pass
        return self

    def prepare(self):
        self.tool = self.options[self.technique](**self.parameters)
        return self

    def start(self, ingredients):
        ingredients = self.tool(ingredients)
        return ingredients