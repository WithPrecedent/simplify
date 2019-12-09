"""
.. module:: filer
:synopsis: base classes for file management
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.creator.options import SimpleOptions
from simplify.library.utilities import listify


@dataclass
class SimpleFiler(ABC):
    """Base class for storing and creating file paths."""

    """ Public Methods """

    def load(self,
            name: Optional[str] = None,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> None:
        """Loads object from file into the subclass attribute 'name'.

        For any arguments not passed, default values stored in the shared
        Filer instance will be used based upon the current 'stage' of the
        siMpLify project.

        Args:
            name (Optional[str]): name of attribute for the file contents to be
                stored. Defaults to None.
            file_path (Optional[str]): a complete file path for the file to be
                loaded. Defaults to None.
            folder (Optional[str]): a path to the folder where the file should
                be loaded from (not used if file_path is passed). Defaults to
                None.
            file_name (Optional[str]): contains the name of the file to be
                loaded without the file extension (not used if file_path is
                passed). Defaults to None.
            file_format (Optional[str]): name of file format in
                filer.extensions. Defaults to None.

        """
        setattr(self, name, self.filer.load(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format))
        return self

    def save(self,
            variable: Optional[Union['SimpleSimpleCodex', str]] = None,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> None:
        """Exports a variable or attribute to disk.

        If 'variable' is not passed, 'self' will be used.

        For other arguments not passed, default values stored in the shared
        filer instance will be used based upon the current 'stage' of the
        siMpLify project.

        Args:
            variable (Optional[Union['SimpleSimpleCodex'], str]): a python object
                or a string corresponding to a subclass attribute which should
                be saved to disk. Defaults to None.
            file_path (Optional[str]): a complete file path for the file to be
                saved. Defaults to None.
            folder (Optional[str]): a path to the folder where the file should
                be saved (not used if file_path is passed). Defaults to None.
            file_name (Optional[str]): contains the name of the file to be saved
                without the file extension (not used if file_path is passed).
                Defaults to None.
            file_format (Optional[str]): name of file format in
                filer.extensions. Defaults to None.

        """
        # If variable is not passed, the subclass instance is saved.
        if variable is None:
            variable = self
        # If a string, 'variable' is converted to a local attribute with the
        # string as its name.
        else:
            try:
                variable = getattr(self, variable)
            except TypeError:
                pass
        self.filer.save(
            variable = variable,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format)
        return self
