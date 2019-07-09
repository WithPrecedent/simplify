
from dataclasses import dataclass

from .steps import Sow, Reap, Clean, Bundle, Deliver


@dataclass
class Almanac(object):
    """Defines rules for sowing, reaping, cleaning, bundling, and delivering
    data as part of the siMpLify Harvest subpackage.

    Attributes:
        menu: an instance of Menu.
        inventory: an instance of Inventory.
        order: a list of Harvest steps to complete.
        auto_prepare: a boolean value as to whether the prepare method should
            be called when the class is instanced.
    """
    menu : object
    inventory : object
    order : object = None
    auto_prepare : bool = True

    def __post_init__(self):
        self._set_defaults()
        return self

    def _check_sections(self):
        if not self.sections:
            self.sections = self.default_sections
        return self

    def _check_steps(self):
        if not hasattr(self, 'steps') or not self.steps:
            self.steps = self.default_steps
        return self

    def _conform_datatypes(self):
        """Adjusts some of the siMpLify-specific datatypes to the appropriate
        datatype based upon the step of the Almanac.
        """
        for section, datatype in self.sections.items():
            if self.step in ['reaper', 'cleaner']:
                if datatype in ['category', 'encoder', 'interactor']:
                    self.sections[section] = str
            elif self.step in ['bundler', 'delivery']:
                if datatype in ['list', 'pattern']:
                    self.sections[section] = 'category'
        return self

    def _localize(self):
        """Injects parameters and 'general' section of menu into each step
        base class and relevant menu settings into this class instance."""
        for step, step_class in self.steps.items():
            if (step_class.name in self.menu.config
                    and not step_class.parameters):
                setattr(step_class, 'parameters', self.menu.config[step])
            self.menu.localize(instance = step_class, sections = ['general'])
            for algorithm in step_class.techniques.values():
                algorithm.inventory = self.inventory
        self.menu.localize(instance = self, sections = ['general'])
        return self

    def _set_defaults(self):
        self.default_steps = {'sower' : Sow,
                              'reaper' : Reap,
                              'cleaner' : Clean,
                              'bundler' : Bundle,
                              'delivery' : Deliver}
        self.default_sections = {}
        for step, step_class in self.steps.items():
            for option in step_class.options.keys():
                setattr(self, 'default_' + option, {})
        return self

    def add_sections(self, sections):
        self._check_sections()
        self.sections.update(sections)
        return self

    def prepare(self):
        self._check_steps()
        self._check_sections()
        for step in self.order:
            for option in self.steps[step].options.keys():
                if not hasattr(self, option):
                    setattr(self, option, 'default_' + option)
        return self

    def start(self, step):
        self.step = step
        self._conform_datatypes()
        return self