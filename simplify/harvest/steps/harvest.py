
from dataclasses import dataclass
import os

from ...implements import ReOrganize, ReSearch
from ...managers.step import Step

@dataclass
class Harvest(Step):

    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'harvester'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _prepare_organizer(self, almanac):
        file_path = os.path.join(self.inventory.organizers,
                                 self.parameters['file_name'])
        parameters = ({'file_path' : file_path,
                       'out_prefix' : 'section_'})
        self.algorithm = self.options[self.technique](**parameters)
        return self

    def _prepare_parsers(self, almanac):
        file_path = os.path.join(self.inventory.parsers,
                                 self.parameters['section'] + '.csv')
        parameters = {'file_path' : file_path,
                      'out_type' : self.parameters['out_type'],
                      'out_prefix' : self.parameters['section'] + '_'}
        self.algorithm = self.options[self.technique](**parameters)
        return self

    def _set_defaults(self):
        self.options = {'organizer' : ReOrganize,
                        'parser' : ReSearch}
        return self

    def _start_organizer(self, ingredients):
        ingredients.df, ingredients.source = self.algorithm.match(
                df = ingredients.df, source = ingredients.source)
        return ingredients

    def _start_parser(self, ingredients):
        source = getattr(ingredients, self.parameters['source'])
        ingredients.df = self.algorithm.match(df = ingredients.df,
                                              source = ingredients.source)
        return ingredients

    def prepare(self):
        getattr(self, '_prepare_' + self.technique)()
        return self

    def start(self, ingredients):
        ingredients = getattr(self, '_start_' + self.technique)(ingredients)
        return ingredients