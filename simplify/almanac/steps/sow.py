
from dataclasses import dataclass
import os
import requests

from ..almanac_step import AlmanacStep


@dataclass
class Sow(AlmanacStep):

    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        self.options = {'convert' : Convert,
                        'download' : Download,
                        'scrape' : Scrape,
                        'split' : Split}
        self.needed_parameters = {'convert' : ['file_in', 'file_out',
                                                 'method'],
                                  'download' : ['file_url', 'file_name'],
                                  'scrape' : ['file_url', 'file_name'],
                                  'split' : ['in_folder', 'out_folder',
                                                'method']}
        if self.technique in ['split']:
            self.import_folder = 'raw'
            self.export_folder = 'interim'
        else:
            self.import_folder = 'external'
            self.export_folder = 'external'
        return self

    def start(self, ingredients):
        self.algorithm.start(ingredients)
        return ingredients

@dataclass
class Convert(object):
    """Converts external data to usable form."""
    file_in : str = ''
    file_out : str = ''
    method : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def _make_path(self, file_name):
        file_path = os.path.join(self.inventory.external, file_name)
        return file_path

    def prepare(self):
        self.file_path_in = self.make_path(self.file_in)
        self.file_path_out = self.make_path(self.file_out)
        return self

    def start(self, ingredients):
        converted = self.method(file_path = self.file_path_in)
        self.inventory.save_df(converted, file_path = self.file_path_out)
        return self

@dataclass
class Download(object):
    """Downloads online data for use by siMpLify."""
    core_data : bool = True
    file_name : str = ''
    file_url : str = ''

    def __post_init__(self):
        super().__post_init__()
        return self

    def start(self, ingredients):
        """Downloads file from a URL if the file is available."""
        file_path = os.path.join(self.inventory.external,
                                 self.file_name)
        file_response = requests.get(self.file_url)
        with open(file_path, 'wb') as file:
            file.write(file_response.content)
        return self

@dataclass
class Scrape(object):

    core_data : bool = True
    file_name : str = ''
    file_url : str = ''
    method : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def start(self, ingredients):
        file_path = os.path.join(self.inventory.external, self.file_name)
        return self

@dataclass
class Split(object):

    in_folder : str = ''
    out_folder : str = ''
    method : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def start(self, ingredients):
        self.method(in_folder = self.in_folder,
                    out_folder = self.out_folder)
        return self