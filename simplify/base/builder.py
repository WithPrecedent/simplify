"""
.. module:: content
:synopsis: base classes for building composite objects
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import Container
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.base.options import SimpleOptions
from simplify.core.utilities import listify


@dataclass
class SimpleDirector(object):
    """Base class for directing the building of objects.

    Args:
        idea ('Idea'): an instance of Idea with user settings.
        content (Optional[Union['Content', List['Content']]]):
            subclasses, not instances, of Content. Defaults to None.
        outline (Optional['Outline']): instance containing information
            needed to build the desired objects. Defaults to None.
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
    content: Optional[Union['Content', List['Content']]] = None
    outline: Optional['Outline'] = None
    auto_publish: Optional[bool] = True
    name: Optional[str] = None

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

    def _initialize_content(self,
            content: Union['Content', List['Content']]) -> None:
        """Instances all passed 'content' and returns instanced list.

        Args:
            content (Optional[Union['Content'], List['Content']]):
                instance(s) of Content subclass. Defaults to None.

        """
        instanced_content = []
        for content in listify(content):
            instanced_content.append(content(idea = self.idea))
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
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Injects attributes from Idea instance, if values exist."""
        self = self.idea.apply(instance = self)
        for content in self.content:
            content.publish()
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
        for content in listify(self.content):
            content.publish()
            # Validates 'content', if possible.
            try:
                for component in content.components:
                    method = '_'.join(['_build', component])
                    if not hasattr(content, method):
                        raise NotImplementedError(' '.join([
                            content.name,
                            'requires build method for every component']))
            except AttributeError:
                pass
        return self

    def apply(self,
            composite: 'SimpleComposite',
            outline: Optional['Outline'],
            **kwargs) -> 'SimpleComposite':
        """Builds and returns SimpleComposite object.

        If subclass instances provide their own methods, they should incorporate
        or call the code below.

        Args:
            composite ('SimpleComposite'): class, not instance, of
                SimpleComposite subclass to return with components added.
            outline (Optional['Outline']): instance containing
                information needed to build the desired objects. Defaults to
                None.
            kwargs (Dict[str, Any]): keyword arguments to pass to content.

        """
        if outline is None:
            outline = self.outline
        elif self.outline is None:
            self.outline = outline
        components = {}
        for content in self.content:
            components[content.name] = content.apply(
                outline = outline, **kwargs)
        return composite(components = components)



@dataclass
class Outline(Container):
    """Base class for object construction instructions."""

    """ Required ABC Methods """

    def __contains__(self, attribute: str) -> bool:
        """Returns whether attribute exists in class instance.

        Args:
            attribute (str): name of attribute to check.

        Returns:
            bool: whether the attribute exists and is not None.

        """
        return hasattr(self, attribute) and getattr(self, attribute) is not None
