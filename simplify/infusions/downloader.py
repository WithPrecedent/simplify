
from dataclasses import dataclass
from functools import wraps
from inspect import getfullargspec
import os
import requests

from .._library import Library


@dataclass
class Downloader(Library):
    """Downloads online data for use by siMpLify."""
    filer : object = None
    file_paths : str = ''
    file_names : str = ''
    file_urls : str = ''

    def __post_init__(self):
        super().__post_init__()
        self._set_defaults()
        self.download()
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
    def download(self, file_paths = None, file_urls = None):
        self._check_lengths()
        file_pairs = zip(self._listify(file_paths), self._listify(file_urls))
        for file_path, file_url in file_pairs.items():
            self._file_download(file_path = file_path, file_url = file_url)
        return self