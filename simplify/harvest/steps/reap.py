
from dataclasses import dataclass
import os

from ...implements import ReOrganize, ReSearch
from ...managers.step import Step

@dataclass
class Reap(Step):

    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'reaper'

    def __post_init__(self):

        super().__post_init__()
        return self

    def _prepare_keywords(self, almanac):
        for section in almanac.keywords:
            file_path = os.path.join(self.inventory.keywords,
                                     section + '.csv')
            parameters = {'file_path' : file_path,
                          'out_type' : self.sections[section],
                          'out_prefix' : section + '_'}
            self.techniques.update(
                    {section : self.options['keywords'](parameters)})
        return self

    def _prepare_organizer(self, almanac):
        file_path = os.path.join(self.inventory.organizers,
                                 almanac.organizer_file)
        parameters = ({'file_path' : file_path,
                       'out_prefix' : 'section_'})
        self.techniques.update(
                {'organizer' : self.options['organizer'](parameters)})
        return self

    def _set_defaults(self):
        self.options = {'organizer' : ReOrganize,
                        'keywords' : ReSearch}
        return self

    def start(self, ingredients, almanac):
        for technique, algorithm in self.techniques.items():
            if technique in ['organizer']:
                ingredients.df, ingredients.source = technique.match(
                        ingredients.df, ingredients.source)
            else:
                ingredients.df = technique.match(ingredients.df,
                                                 ingredients.source)
        return ingredients