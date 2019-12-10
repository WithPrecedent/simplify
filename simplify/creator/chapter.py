"""
.. module:: chapter
:synopsis: composite tree class for storing iterables
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify import creator
from simplify.creator.codex import SimpleCodex
from simplify.library.utilities import listify


@dataclass
class Chapter(SimpleCodex):
    """Iterator for a siMpLify process.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        techniques (Optional[List[str], str]): ordered list of techniques to
            use. Each technique should match a key in 'options'. Defaults to
            None.
        options (Optional[Union['Options', Dict[str, Any]]]): allows
            setting of 'options' property with an argument. Defaults to None.
        metadata (Optional[Dict[str, Any]], optional): any metadata about
            the Chapter. In projects, 'number' is automatically a key
            created for 'metadata' to allow for better recordkeeping.
            Defaults to None.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. Defaults to True.
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.
        export_folder (Optional[str]): attribute name of folder in 'filer' for
            serialization of subclasses to be saved. Defaults to 'chapter'.

    """
    name: Optional[str] = 'chapter'
    techniques: Optional[Union[List[str], str]] = None
    options: (Optional[Union['Options', Dict[str, Any]]]) = None
    metadata: Optional[Dict[str, Any]] = None
    auto_publish: Optional[bool] = True
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'

    def __post_init__(self) -> None:
        self.proxies = {'parent': 'book', 'children': 'pages', 'child': 'page'}
        super().__post_init__()
        return self

    """ Private Methods """
    
    def _draft_techniques(self):
        if not self.techniques:
            self.techniques = {}
        return self
    
    """ Core siMpLify Methods """
        
    def publish(self, data: Optional[object] = None) -> None:
        """Required method which applies methods to passed data.

        Subclasses should provide their own 'publish' method.

        Args:
            data (Optional[object]): an optional object needed for the method.

        """
        if data is None:
            try:
                data = self.ingredients
            except AttributeError:
                pass
        self.options.publish(
            techniques = self.techniques,
            data = data)
        return self

    def apply(self, data: Optional[object], **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Subclasses should provide their own 'apply' method, if needed.

        Args:
            data (object): data object for methods to be applied.

        """
        if data is None:
            try:
                data = self.ingredients
            except AttributeError:
                pass
        for technique in self.techniques:
            data = self.options[technique].options.apply(
                key = technique,
                data = data,
                **kwargs)
        return data