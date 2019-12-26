"""
.. module:: book
:synopsis: subpackage base classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Collection
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.base import SimpleManuscript


@dataclass
class Book(SimpleManuscript):

    project: 'Project'
    name: Optional[str] = None
    steps: 'SimpleSequence' = field(default_factory = list)
    options: 'SimpleOptions' = field(default_factory = dict)
    chapters: List['Chapter'] = field(default_factory = list)
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'


@dataclass
class Chapter(SimpleManuscript):

    book: 'Book'
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'chapter'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self


@dataclass
class Page(SimpleManuscript):

    book: 'Book'
    algorithm: 'Algorithm'
    parameters: 'Parameters'
    name: Optional[str] = None
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'chapter'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self