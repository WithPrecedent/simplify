"""
.. module:: author
:synopsis: content builder
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""
from abc import ABC
from abc import abstractmethod
from collections.abc import Container
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.options import SimpleOptions
from simplify.core.utilities import listify


@dataclass
class Author(ABC):
    """Creates Books.
    
    Author subclasses direct the creation of siMpLify classes in the following 
    manner.
    
        Idea -> Options -> Outline -> Content -> Page -> Chapter -> Book

    Args:
        idea ('Idea'): an instance of Idea with user settings.
        outline (Optional['Outline']): instance containing information
            needed to build the desired objects. Defaults to None.
        content (Optional[Union['Content', List['Content']]]): subclasses, not 
            instances, of Content. Defaults to None.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_publish to have an effect,
            'outline' and 'content' must also be passed. Defaults to True.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    idea: 'Idea'
    outline: Optional['Outline'] = None
    content: Optional[Union['Content', List['Content']]] = None
    auto_publish: Optional[bool] = True
    name: Optional[str] = None
    
    """
    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supoorted file type with settings for an Idea instance is
            located.
        library (Optional[Union['Library', str]]): an instance of
            library or a string containing the full path of where the root
            folder should be located for file output. A library instance
            contains all file path and import/export methods for use throughout
            the siMpLify package. Default is None.
        ingredients (Optional[Union['Ingredients', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Ingredients, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Library instance, a
            DataFrame, a Series, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Ingredients instance. Default is None.
        steps (Optional[Union[List[str], str]]): ordered names of Book
            subclasses to include. These names should match keys in the
            'options' attribute. If using the Idea instance settings, this
            argument should not be passed. Default is None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_publish to have an effect,
            'ingredients' must also be passed. Defaults to True.
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.
        export_folder (Optional[str]): attribute name of folder in 'library' for
            serialization of subclasses to be saved. Defaults to 'book'.

    """
    idea: Union['Idea', str]
    library: Optional[Union['Library', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        str]] = None
    steps: Optional[Union[List[str], str]] = None
    name: Optional[str] = 'simplify'
    auto_publish: Optional[bool] = True
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'
    
    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if (self.auto_publish
                and self.outline is not None
                and self.content is not None):
            self.publish()
        return self

    def _draft_content(self, 
            content: Union['Content', List['Content']]) -> None:
        """Instances all 'content' and returns instanced list.

        Args:
            content (Optional[Union['Content', List['Content']]):
                instance(s) of Content subclass. Defaults to None.

        """
        instanced_content = []
        for item in listify(content):
            # Checks to see if class has already been instanced.
            if not isinstance(item, Content):
                instanced_item = item()
                instanced_item.author = self
                instanced_content.append(instanced_item)
        return instanced_content

    """ Public Methods """

    def add_content(self,
            content: Union['Content', List['Content']],
            replace_content: Optional[bool] = False) -> None:
        """Adds Content classes to 'content' attribute.

        Args:
            content (Union['Content'], List['Content']):
                subclass(es), not instance(s), of Content.
            replace_content (Optional[bool]): whether to replace existing
                'content' (True) or add them to existing 'content' (False).

        """
        if replace_content or self.content is None:
            self.content = listify(content)
        else:
            self.content.extend(listify(content))
        self.content = self._draft_content(content = self.content)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Injects attributes from Idea instance, if values exist."""
        self = self.idea.apply(instance = self)
        self.content = self._draft_content(content = self.content)
        return self

    def publish(self,
            content: Optional[Union['Content'],
                               List['Content']] = None,
            replace_content: Optional[bool] = False) -> None:
        """Validates content.

        If subclass instances provide their own methods, they should incorporate
        or call the code below.

        Args:
            content (Optional[Union['Content'], List['Content']]):
                instance(s) of Content subclass. Defaults to None.
            replace_content (Optional[bool]): if 'content' is passed, whether
                to replace existing 'content' (True) or add them to existing
                'content'.

        """
        if content is not None:
            self.add_content(
                content = content,
                replace_content = replace_content)
        for item in listify(self.content):
            item.publish()
            # Validates 'content', if possible.
            try:
                for component in item.components:
                    method = '_'.join(['_build', component])
                    if not hasattr(item, method):
                        raise NotImplementedError(' '.join([
                            content.name,
                            'requires build method for every component']))
            except AttributeError:
                pass
        return self

    def apply(self,
            page: 'Page',
            outline: Optional['Outline'],
            **kwargs) -> 'SimpleComposite':
        """Builds and returns Page object.

        If subclass instances provide their own methods, they should incorporate
        or call the code below.

        Args:
            page ('Page'): class, not instance, of page subclass to return with 
                components added.
            outline (Optional['Outline']): instance containing information 
                needed to build the desired objects. Defaults to None.
            kwargs (Dict[str, Any]): keyword arguments to pass to content.

        """
        if outline is None:
            outline = self.outline
        elif self.outline is None:
            self.outline = outline
        components = {}
        for component in self.content:
            components[component.name] = component.apply(
                outline = outline, **kwargs)
        return page(components = components)

