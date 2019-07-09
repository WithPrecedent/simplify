
from dataclasses import dataclass
import warnings

from ..inventory import Inventory
from ..menu import Menu


@dataclass
class Planner(object):
    """Parent class for Cookbook and Harvest to provide shared methods for
    creating data science workflows.
    """
    menu : object
    inventory : object = None

    def __post_init__(self):
        """Implements basic settings for Planner subclasses."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        self._check_menu()
        self._check_inventory()
        self.menu.localize(instance = self, sections = ['general', self.name])
        # Calls prepare method if subclass has an auto_prepare attribute that
        # is set to True.
        if hasattr(self, 'auto_prepare') and self.auto_prepare:
            self.prepare()
        return self

    def _check_inventory(self):
        """Adds a Inventory instance with default menu if one is not passed
        when a Planner subclass is instanced.
        """
        if not self.inventory:
            self.inventory = Inventory(menu = self.menu)
        return self

    def _check_menu(self):
        """Loads menu from an .ini file if a string is passed to menu instead
        of a menu instance.
        """
        if isinstance(self.menu, str):
            self.menu = Menu(file_path = self.menu)
        return self