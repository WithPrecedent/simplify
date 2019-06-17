
from dataclasses import dataclass
import os
import re

from .implement import Implement
from .matcher import ReMatch


@dataclass
class Divider(Implement):

    origins : object = None
    techniques : object = None
    regexes : object = None
    prefix : str = 'section_'
    file_path : str = ''

    def __post_init__(self):
        self.default_origins = {}
        self.default_techniques = {}
        self.default_regexes = {}
        self.default_file_path = os.path.join('dividers', 'dividers.csv')
        return self

    def add_technique(self, techniques, funcs):
        new_techniques = zip(self._listify(techniques),
                             self._listify(funcs))
        for technique, regex, func in new_techniques.items():
            if not self.techniques:
                self.techniques = {}
            self.options.update({technique, func})
            setattr(self, func.__name__, func)
        return self

    def _no_breaks(self, variable):
        """Removes line breaks and replaces them with single spaces."""
        variable.str.replace('[a-z]-\n', '')
        variable.str.replace('\n', ' ')
        return variable

    def create(self):
        self.regexes = ReMatch(file_path = self.file_path,
                               reverse_dict = True).expressions
        return self

    def extract(self, section, regex):
        matched = self.find(section, regex)
        origin = getattr(self, self.origins[section])
        origin.replace(matched, '')
        return matched

    def extract_all(self, section, regex):
        matched = self.find_all(section, regex)
        for string in matched:
            origin = getattr(self, self.origins[section])
            origin.replace(string, '')
        return matched

    def find(self, section, regex):
        if re.search(regex, self.origins[section]):
            matched = re.search(regex, self.origins[section]).group(0).strip()
        else:
            matched = ''
        return matched

    def find_all(self, section, regex):
        if re.search(regex, self.origins[section]):
            matched = re.findall(regex, self.origins[section])
        else:
            matched = []
        return matched

    def initialize(self):
        if not self.origins:
            self.origins = self.default_origins
        if not self.techniques:
            self.techniques = self.default_techniques
        if not self.regexes:
            if not self.file_path:
                self.file_path = self.default_file_path
            self.create()
        return self

    def iterate(self, df):
        for section, regex in self.regexes.items():
            column = self.prefix + section
            df[column] = self._no_breaks(
                    self.techniques[section](section, regex))
        return df