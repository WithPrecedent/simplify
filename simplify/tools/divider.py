
from dataclasses import dataclass
import os
import re

from .matcher import ReMatch


@dataclass
class Divider(object):

    sources : object = None
    techniques : object = None
    prefix : str = 'section_'
    grid : object = None
    file_path : str = ''
    file_name : str = ''

    def __post_init__(self):
        super().__post_init__()
        self.options = {'find' : self._find,
                        'find_all' : self._find_all,
                        'extract' : self._extract,
                        'extract_all' : self._extract_all}
        return

    def _set_dividers(self):
        if not self.grid:
            if self.file_path:
                df = ReMatch(file_path = self.file_path,
                             reverse_dict = True).expressions
            elif self.file_name:
                self.file_path = os.path.join(self.filer.data, self.file_name)
                df = ReMatch(file_path = self.file_path,
                             reverse_dict = True).expressions
            else:
                error = 'Divider requires file_path, file_name or regexes'
                raise AttributeError(error)
            self.grid = df.todict
        return self

    def add_technique(self, techniques, regexes, func):
        new_techniques = zip(self._listify(techniques),
                             self._listify(regexes),
                             self._listify(func))
        for technique, regex, fun in new_techniques.items():
            self.options.update({technique, regex})
            setattr(self, fun.__name__, fun)
        return self

    def create(self):
        self._set_dividers()
        return self

    def extract(self, name, regex):
        matched = self.find(name, regex)
        self.sources[name].replace(matched, '')
        return matched, self.sources[name]

    def extract_all(self, name, regex):
        matched = self.find_all(name, regex)
        self.sources[name].replace(matched, '')
        return matched

    def find(self, name, regex):
        if re.search(regex, self.sources[name]):
            matched = re.search(regex, self.sources[name]).group(0)
        else:
            matched = ''
        return matched

    def find_all(self, name, regex):
        if re.search(regex, self.sources[name]):
            matched = re.findall(regex, self.sources[name])
        else:
            matched = ''
        return matched

    def iterate(self, df):
        for section, regex in self.grid.items():
            df[section] = self.options[self.techniques[section]](section,
                                                                 regex)
        return df