"""
.. module:: loader
:synopsis: lazy loading made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import dataclasses
import importlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import simplify
from simplify import core


@dataclasses.dataclass
class SimpleLoader(core.SimpleComponent):
    """Base class for lazy loaders.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared Idea instance, 'name'
            should match the appropriate section name in that Idea instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        default_module (Optional[str]): a backup name of module where object to
            use is located (can either be a siMpLify or non-siMpLify module).
            Defaults to 'simplify.core'.

    """
    name: Optional[str] = None
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.core')
    default_module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.core')

    """ Public Methods """

    def load(self, attribute: str) -> object:
        """Returns object named in 'attribute'.

        If 'attribute' is not a str, it is assumed to have already been loaded
        and is returned as is.

        The method searches both 'module' and 'default_module' for the named
        'attribute'.

        Args:
            attribute (str): name of local attribute to load from 'module' or
                'default_module'.

        Returns:
            object: from 'module' or 'default_module'.

        """
        # If 'attribute' is a string, attempts to load from 'module' or, if not
        # found there, 'default_module'.
        if isinstance(getattr(self, attribute), str):
            try:
                return getattr(
                    importlib.import_module(self.module),
                    getattr(self, attribute))
            except (ImportError, AttributeError):
                try:
                    return getattr(
                        importlib.import_module(self.default_module),
                        getattr(self, attribute))
                except (ImportError, AttributeError):
                    raise ImportError(
                        f'{getattr(self, attribute)} is neither in \
                        {self.module} nor {self.default_module}')
        # If 'attribute' is not a string, it is returned as is.
        else:
            return getattr(self, attribute)