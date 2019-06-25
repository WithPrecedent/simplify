
from dataclasses import dataclass
import os

from .stage import Stage
from ...implements.retool import ReMatch


@dataclass
class Thresh(Stage):

    techniques : object = None
    machines : object = None
    machines_path = object = None
    add_folder_to_file_names : bool = True

    def __post_init__(self):
        self._initialize()
        return

    def _parse_imported_techniques(self):
        threshers = self.combine_lists(self.machines_df['names'].tolist(),
                                       self.machines_df['out_types'].tolist(),
                                       self.machines_df['origins'].tolist(),
                                       self.machines_df['files'].tolist())
        self.add_machines(threshers)
        return self

    def add_machines(self, machines):
        for name, out_type, origin, file in machines:
            if self.add_folder_to_file_names:
                file = os.path.join(self.inventory.threshers, file)
            machine = ReMatch(file_path = file,
                              in_col = origin,
                              out_type = out_type,
                              out_prefix = name + '_')
            self.machines.update({name : machine})
        return self

    def create(self, df):
        """Applies threshing machines to dataframe or series."""
        for name, machine in self.machines.items():
            df = machine.match()
        return df

    def prepare(self):
        """If machines do not currently exist, lists of names, techniques,
        origins, and files are combined to create machines.
        """
        if self.machines_path:
            self.import_techniques()
            self._parse_imported_techniques()
        return self