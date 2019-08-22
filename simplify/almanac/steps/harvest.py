
from dataclasses import dataclass
import os

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

    def _prepare_organize(self, key):
        file_name = 'organizer_' + key + '.csv'
        self.parameters = {'file_path' : os.path.join(self.inventory.external,
                                                      file_name)}
        return self

    def _prepare_parse(self, key):
        file_name = 'parser_' + key + '.csv'
        self.parameters = {'file_path' : os.path.join(self.inventory.external,
                                                      file_name)}
        return self

    def _set_defaults(self):
        self.options = {'organize' : ReTool,
                        'parse' : ReTool}
        return self

    def _set_initial_columns(self, algorithm):
        self.columns = list(algorithm.expressions.values())
        prefix = algorithm.matcher.section
        self.columns = [prefix + '_' + column for column in self.columns]
        return self

    def _start_organize(self, ingredients, algorithm):
        ingredients.df, ingredients.source = algorithm.start(
                df = ingredients.df, source = ingredients.source)
        return ingredients

    def _start_parse(self, ingredients, algorithm):
        ingredients.df = algorithm.start(df = ingredients.df,
                                         source = ingredients.source)
        return ingredients

    def prepare(self):
        for key in self.parameters:
            if hasattr(self, '_prepare_' + self.technique):
                getattr(self, '_prepare_' + self.technique)(key = key)
            else:
                getattr(self, '_prepare_generic_list')(key = key)
            algorithm = self.options[self.technique](**self.parameters)
            if self.technique == 'organize':
                self._set_initial_columns(algorithm)
            self.algorithms.append(algorithm)
        return self