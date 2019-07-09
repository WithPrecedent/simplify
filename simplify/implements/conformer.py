
from dataclasses import dataclass


@dataclass
class Conformer(object):

    menu : object
    inventory : object
    step : str = ''
    auto_prepare : bool = True

    def __post_init__(self):
        self.menu.localize(instance = self, sections = ['general'])
        if self.auto_prepare:
            self.prepare()
        return self

    def _conform_datatypes(self):
        """Adjusts some of the siMpLify-specific datatypes to the appropriate
        datatype based upon the step of the Almanac.
        """
        for prefix, datatype in self.prefixes.items():
            if self.step in ['reap', 'clean']:
                if datatype in ['category', 'encoder', 'interactor']:
                    self.prefixes[prefix] = str
            elif self.step in ['bundle', 'deliver']:
                if datatype in ['list', 'pattern']:
                    self.prefixes[prefix] = 'category'
        return self

    def prepare(self, step = None):
        if step:
            self.step = step
        return self

    def start(self, almanac):
        if self.verbose:
            print('Conforming almanac')

        return almanac