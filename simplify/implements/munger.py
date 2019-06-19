
from dataclasses import dataclass
import os

from ..countertop import Countertop
from .matcher import ReMatch


@dataclass
class Munger(Countertop):

    origins : object = None
    techniques : object = None
    munger_files : object = None

    def __post_init__(self):
        self.options = {'boolean' : self._create_boolean,
                        'float' : self._create_float,
                        'int' : self._create_int,
                        'list' : self._create_list,
                        'str' : self._create_str,
                        'pattern' : self._create_pattern}
        return

    def _create_boolean(self, file_path, section):
        rematch = ReMatch(file_path = file_path,
                          in_col = self.origins[section],
                          out_type = bool,
                          out_prefix = section + '_')
        return rematch

    def _create_float(self, file_path, section):
        rematch = ReMatch(file_path = file_path,
                          in_col = self.origins[section],
                          out_type = float,
                          out_prefix = section + '_')
        return rematch

    def _create_int(self, file_path, section):
        rematch = ReMatch(file_path = file_path,
                          in_col = self.origins[section],
                          out_type = int,
                          out_prefix = section + '_')
        return rematch

    def _create_list(self, file_path, section):
        rematch = ReMatch(file_path = file_path,
                          in_col = self.origins[section],
                          out_type = list,
                          out_col = section)
        return rematch

    def _create_str(self, file_path, section):
        rematch = ReMatch(file_path = file_path,
                          in_col = self.origins[section],
                          out_type = str,
                          out_prefix = section + '_')
        return rematch

    def _create_pattern(self, file_path, section):
        rematch = ReMatch(file_path = file_path,
                          in_col = self.origins[section],
                          out_type = 'pattern',
                          out_prefix = section + '_')
        return rematch


    def add_technique(self, techniques, funcs):
        new_techniques = zip(self._listify(techniques),
                             self._listify(funcs))
        for technique, regex, func in new_techniques.items():
            self.options.update({technique, func})
            setattr(self, func.__name__, func)
        return self

    def create(self):
        if not self.techniques:
            self.techniques = {}
        for section, munger_file in self.munger_files.items():
            file_path = os.path.join(self.paths.mungers, munger_file)
            if section in self.origins:
                rematch = ReMatch(file_path = file_path,
                                  in_col = self.origins[section],
                                  out_type = self.data_types[section],
                                  out_prefix = section + '_')
            else:
                rematch = ReMatch(file_path = file_path,
                                  out_type = self.data_types[section],
                                  out_prefix = section + '_')
            self.techniques.update({section : rematch})
        return self

    def iterate(self, df):
        for section, munger in self.techniques.items():
            df[section] = self.options[self.techniques[section]](section,
                                                                 munger)
        return df