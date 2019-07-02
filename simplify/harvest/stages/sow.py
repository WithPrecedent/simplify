
from dataclasses import dataclass
from functools import wraps
from inspect import getfullargspec
import os
import requests

from .stage import Stage, Technique


@dataclass
class Sow(Stage):

    technique : str = ''
    parameters : object = None
    name : str = 'sow'
    auto_prepare : bool = True

    def __post_init__(self):
        self.techniques = {'convert' : Convert,
                           'download' : Download,
                           'scrape' : Scrape,
                           'split' : Split}
        super().__post_init__()
        return self

    def start(self):
        kwargs = self.parameters
        self.techniques[self.technique](**kwargs)
        return self

@dataclass
class Convert(Technique):

    inventory : object = None
    file_path_in : str = ''
    file_path_out : str = ''
    technique : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

@dataclass
class Download(Technique):
    """Downloads online data for use by siMpLify."""

    inventory : object = None
    file_name : str = ''
    file_url : str = ''
    technique : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def _file_download(self, file_path, file_url):
        """Downloads file from a URL if the file is available."""
        file_response = requests.get(file_url)
        with open(file_path, 'wb') as file:
            file.write(file_response.content)
        return self

    def _set_defaults(self):
        """Sets file_paths based upon arguments passed when class is instanced.
        """
        if not self.file_paths:
            if self.filer:
                if self.file_names:
                    self.file_paths = []
                    for file_name in self._listify(self.file_names):
                        self.file_paths.append(os.path.join(
                                self.filer.externals, self.file_name))
                else:
                    error = 'Downloader requires file_names or file_paths'
                    raise AttributeError(error)
            else:
                error = 'Downloader requires Filer instance or file_paths'
                raise AttributeError(error)
        return self

    def check_path(func):
        """Decorator which uses default file_path if file_names or file_paths
        are not included in arguments.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            argspec = getfullargspec(func)
            unpassed_args = argspec.args[len(args):]
            if 'file_path' in argspec.args and 'file_path' in unpassed_args:
                kwargs.update({'file_path' : getattr(self, self.file_path)})
            if 'file_url' in argspec.args and 'file_url' in unpassed_args:
                kwargs.update({'file_url' : getattr(self, self.file_url)})
            return func(self, *args, **kwargs)
        return wrapper

    @check_path
    def start(self, file_paths = None, file_urls = None):
        self._check_lengths()
        file_pairs = zip(self._listify(file_paths), self._listify(file_urls))
        for file_path, file_url in file_pairs.items():
            self._file_download(file_path = file_path, file_url = file_url)
        return self

@dataclass
class Scrape(technique):

    inventory : object = None
    file_name : str = ''
    file_url : str = ''
    technique : object = None

    def __post_init__(self):
        super().__post_init__()
        self._set_defaults()
        self.scrape()
        return self

    def _file_scrape(self, file_path, file_url):
        """Scrapes file or data from a URL if the file is available."""
        return self

    def _set_defaults(self):
        """Sets file_paths based upon arguments passed when class is instanced.
        """
        if not self.file_paths:
            if self.filer:
                if self.file_names:
                    self.file_paths = []
                    for file_name in self._listify(self.file_names):
                        self.file_paths.append(os.path.join(
                                self.filer.externals, self.file_name))
                else:
                    error = 'Downloader requires file_names or file_paths'
                    raise AttributeError(error)
            else:
                error = 'Downloader requires Filer instance or file_paths'
                raise AttributeError(error)
        return self

    def check_path(func):
        """Decorator which uses default file_path if file_names or file_paths
        are not included in arguments.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            argspec = getfullargspec(func)
            unpassed_args = argspec.args[len(args):]
            if 'file_path' in argspec.args and 'file_path' in unpassed_args:
                kwargs.update({'file_path' : getattr(self, self.file_path)})
            if 'file_url' in argspec.args and 'file_url' in unpassed_args:
                kwargs.update({'file_url' : getattr(self, self.file_url)})
            return func(self, *args, **kwargs)
        return wrapper

    @check_path
    def scrape(self, file_paths = None, file_urls = None):
        self._check_lengths()
        file_pairs = zip(self._listify(file_paths), self._listify(file_urls))
        for file_path, file_url in file_pairs.items():
            self._file_scrape(file_path = file_path, file_url = file_url)
        return self

class Split(Technique):

    inventory : object = None
    technique : object = None

    def __post_init__(self):
        super().__post_init__()
        return self