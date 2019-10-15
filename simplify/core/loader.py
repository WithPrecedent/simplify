"""
.. module:: lazy loader
:synopsis: allows lazy (delayed) importation of modules
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict

from simplify.core.base import SimpleClass


@dataclass
class LazyImporter(SimpleClass):
    """Lazily imports modules and packages in 'options' attribute.

    To allow users flexibility in dependency usage and lower memory consumption,
    this class imports objects from specified modules at runtime when needed.
    Module references can be internal or to external dependencies as long as
    the import path is valid.

    To use this class, 'options' should be formatted as follows:
        {name(str): (module_path(str), class_name(str))}
                            or
        {name(str): [module_path(str), class_name(str)]}

    The class also maintains lists of packages from designated packages listed
    in 'tracked_packages'. These lists are created so that specific methods
    can be adapted based upon the package source of a technique chosen.

    This class also converts other internal references in 'options' from strings
    to attributes.

    """
    
    name: str = 'lazy_loader'
    auto_publish: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _load_option(self, instance, name, settings):
        if 'self' == settings[0]:
            imported_option = {name: getattr(instance, settings[1])}
        else:
            self._track_packages(setting = settings[0])
            imported_option = ({name: self.lazily_import(
                module = settings[0], name = settings[1])})
        return imported_option
    
    def _track_packages(self, setting):
        """Adds package name to package source registry during lazy
        importation.

        Args:
            settings(str): name of import path for package.

        """
        for package in self.tracked_packages:
            if package in setting:
                if not self.exists('package' + '_options'):
                    setattr(self, 'package' + '_options', [])
                if package not in getattr(self, 'package' + '_options'):
                    getattr(self, 'package' + '_options').append(package)
        return self

    def _using_option(self, instance, key):
        """Returns whether option is being used."""
        if hasattr(instance, 'lazy_imports') and key in instance.lazy_imports:
            return True
        elif hasattr(instance, 'sequence') and key in instance.sequence:
            return True
        elif hasattr(instance, 'technique') and key in instance.technique:
            return True
        elif hasattr(instance, 'model_type') and key in instance.model_type:
            return True
        else:
            return False

    """ Public Input/Output Methods """

    @staticmethod
    def lazily_import(module, name):
        return getattr(import_module(module), name)

    def load(self, instance, attribute = 'options'):
        """Imports modules at runtime."""
        if not hasattr(instance, 'lazy_import') or instance.lazy_import:
            new_options = {}
            for name, settings in getattr(instance, attribute).items():
                if (self._using_option(instance = instance, key = name)
                        and (isinstance(settings, list)
                             or isinstance(settings, tuple))):
                    new_options.update(self._load_option(
                        instance = instance,
                        name = name, 
                        settings = settings))
            getattr(instance, attribute).update(new_options)
        return instance
    
    """ Core siMpLify Methods """
    
    def draft(self):
        super().draft()
        self.tracked_packages = [
            'sklearn',
            'simplify',
            'xgboost',
            'catboost',
            'tensorflow',
            'pytorch',
            'lightgbm']
        return self
    
    def publish(self):
        super().publish()
        self._inject_base(attribute = 'lazy')
        return self