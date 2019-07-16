
from dataclasses import dataclass
import os
import requests

from ...managers import Step, Technique


@dataclass
class Sow(Step):

    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'sower'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _prepare_converters(self, almanac):
        for key, value in almanac.converters.items():
            source_column = 'section_' + key
            out_column = key
            method = almanac.converter_methods[key]
            self.techniques.update({key : self.options['converters'](
                    source_column = source_column,
                    out_column = out_column,
                    method = method)})
        return self

    def _prepare_downloaders(self, almanac):
        for key, value in almanac.downloaders.items():
            file_path_in =
        return self

    def _prepare_scrapers(self, almanac):
        return self

    def _prepare_splitters(self, almanac):
        return self

    def _set_defaults(self):
        self.options = {'converters' : Convert,
                        'downloaders' : Download,
                        'scrapers' : Scrape,
                        'splitters' : Split}
        return self

    def start(self, ingredients, almanac):
        for technique, algorithm in self.techniques.items():
            algorithm.start(ingredients, almanac)
        return ingredients

@dataclass
class Convert(Technique):

    file_path_in : str = ''
    file_path_out : str = ''
    method : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def start(self, ingredients, almanac):
        converted = self.method(file_path = self.file_path_in)
        self.inventory.save_df(converted, file_path = self.inventory.external)
        return self

@dataclass
class Download(Technique):
    """Downloads online data for use by siMpLify."""
    file_name : str = ''
    file_url : str = ''
    method : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def start(self, ingredients, almanac):
        """Downloads file from a URL if the file is available."""
        file_path = os.path.join(self.almanac.inventory.external,
                                 self.file_name)
        file_response = requests.get(self.file_url)
        with open(file_path, 'wb') as file:
            file.write(file_response.content)
        return self

@dataclass
class Scrape(Technique):

    file_name : str = ''
    file_url : str = ''
    method : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def start(self, ingredients, almanac):
        file_path = os.path.join(self.inventory.external, self.file_name)
        return self

@dataclass
class Split(Technique):

    in_folder : str = ''
    out_folder : str = ''
    method : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def start(self, ingredients, almanac):
        self.algorithm(in_folder = self.in_folder,
                       out_folder = self.out_folder)
        return self