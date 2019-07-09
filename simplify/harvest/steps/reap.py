
from dataclasses import dataclass
import os

from .harvest_step import HarvestStep
from ...implements import ReOrganize, ReSearch


@dataclass
class Reap(HarvestStep):

    options : object = None
    almanac : object = None
    auto_prepare : bool = True
    name : str = 'reaper'

    def __post_init__(self):
        self.default_options = {'organizer' : ReOrganize,
                                'keywords' : ReSearch}
        super().__post_init__()
        return self

    def _prepare_keywords(self):
        for section in self.keywords:
            file_path = os.path.join(self.inventory.keywords,
                                     section + '.csv')
            parameters = {'file_path' : file_path,
                          'out_type' : self.sections[section],
                          'out_prefix' : section + '_'}
            self.techniques.update(
                    {section : self.options['keywords'](parameters)})
        return self

    def _prepare_organizer(self):
        file_path = os.path.join(self.inventory.organizers,
                                 self.organizer_file)
        parameters = ({'file_path' : file_path,
                       'out_prefix' : 'section_'})
        self.techniques.update(
                {'organizer' : self.options['organizer'](parameters)})
        return self


    def start(self, ingredients):
        for technique, algorithm in self.techniques.items():
            if technique in ['organizer']:
                ingredients.df, ingredients.source = technique.match(
                        ingredients.df, ingredients.source)
            else:
                ingredients.df = technique.match(ingredients.df,
                                                 ingredients.source)
        return ingredients