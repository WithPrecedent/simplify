
from dataclasses import dataclass
import os
import re

from .stage import Stage
from ...implements.retool import ReArrange


@dataclass
class Reap(Stage):

    techniques : object = None
    machines : object = None
    machines_path = object = None
    add_folder_to_file_names : bool = True
    prefix : object = None
    remove_from_source : bool = True

    def __post_init__(self):
        self.options = {'extract' : self.extract,
                        'extract_all' : self.extract_all,
                        'find' : self.find,
                        'findall' : self.findall}
        self._initialize()
        return self

    def add_machines(self, techniques, machines):
        new_techniques = zip(self._listify(techniques),
                             self._listify(machines))
        for technique, origin, regex in new_techniques.items():
            if not self.techniques:
                self.techniques = {}
            self.options.update({technique, func})
            setattr(self, func.__name__, func)
        return self

    def create(self, df, sources):
        for name, regex in self.machines.items():
            column = self.prefix + name
            df[column] = self._no_breaks(
                    self.techniques[name](name, regex))
        return df

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

    def prepare(self):
        """If machines do not currently exist, lists of names, techniques,
        origins, and files are combined to create machines.
        """
        if self.machines_path:
            self.import_techniques()
            self._parse_imported_techniques()
        self.regexes = ReArrange(file_path = self.file_path)
        return self