
from dataclasses import dataclass


@dataclass
class Stage(object):
    """Controller class for setting various class attributes depending upon the
    current step in the Harvest or Cookbook.
    """

    def __post_init__(self):
        self.options = ['sow', 'reap', 'clean', 'bundle', 'deliver', 'cook']
        return self

    def advance(self, stage):
        if stage in self.options:
            self.stage = stage
        else:
            error = stage + 'not a recognized step in siMpLify'
            raise KeyError(error)
        return self