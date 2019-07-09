
from dataclasses import dataclass

@dataclass
class Technique(object):

    def __post_init__(self):
        return self

    def prepare(self):
        self.tool = self.techniques[self.technique](**self.parameters)
        return self

    def start(self, ingredients):
        ingredients = self.tool(ingredients)
        return ingredients