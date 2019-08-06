
from dataclasses import dataclass

from ...implements.retool import ReTool
from ..almanac_step import AlmanacStep


@dataclass
class Harvest(AlmanacStep):

    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        self.options = {'organize' : ReTool,
                        'parse' : ReTool}
        return self

    def _start_organizer(self, ingredients):
        ingredients.df, ingredients.source = self.algorithm.start(
                df = ingredients.df, source = ingredients.source)
        return ingredients

    def _start_parser(self, ingredients):
        ingredients.df = self.algorithm.start(df = ingredients.df,
                                              source = ingredients.source)
        return ingredients

    def start(self, ingredients):
        ingredients = getattr(self, '_start_' + self.technique)(ingredients)
        return ingredients