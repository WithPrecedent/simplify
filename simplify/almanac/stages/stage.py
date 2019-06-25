
from dataclasses import dataclass

import pandas as pd

from ..blackacre import Blackacre


@dataclass
class Stage(Blackacre):
    """Parent class for various stages related to the Almanac class in the
    siMpLify package to allow sharing of methods.
    """

    def _check_machines(self):
        """Checks if attribute algorithms exists. If not, returns an empty
        list.
        """
        if not self.machines:
            self.machines = {}
        return self

    def _check_techniques(self):
        """Checks if machines are a tuple. If so, a method is called to convert
        them into the appropriate dictionary for the subclass.
        """
        if isinstance(self.techniques, tuple):
            self.add_machines(self.techniques)
        return self

    def _initialize(self):
        """Initializes stage child class."""
        self._check_techniques()
        self._check_machines()
        self.prepare()
        return self

    def combine_lists(self, *args, **kwargs):
        """Combines lists to create a tuple."""
        return zip(*args, **kwargs)

    def import_machines(self):
        """Imports .csv file containing appropriate columns for machines to
        be constructed.
        """
        self.machine_df = pd.read_csv(self.machines_path)
        return self
