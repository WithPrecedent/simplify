"""
.. module:: author
:synopsis: creator of manuscripts
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify import core
from simplify.core.book import Book
from simplify.core.book import Chapter
from simplify.core.book import Page
from simplify.core.options import ManuscriptOptions
from simplify.core.project import Project
from simplify.core.utilities import listify


@dataclass
class Creator(ABC):
    """Base class for Book, Chapter, and Page creation."""
    
    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Validates passed 'options' argument.
        self._validate_options()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if self.auto_publish:
            self.publish()
        return self

    """ Private Methods """

    def _validate_options(self) -> None:
        """Converts 'options' to proper ManuscriptOptions form."""
        if not self.options:
            self.options = ManuscriptOptions(
                options = globals()['DEFAULT_OPTIONS'],
                parent = self)
        elif isinstance(self.options, Dict):
            self.options = ManuscriptOptions(
                options = self.options, 
                parent = self)
        else:
            self.options = ManuscriptOptions(options = {}, parent = self)
        return self


    def _create_filer(self, manuscript_type: str) -> 'SimpleFiler':
        """Returns SimpleFiler object appropriate to 'manuscript_type'.

        Args:
            manuscript_type (str): either 'book', 'chapter', or 'page'.

        Returns:
            'SimpleFiler' with settings for specific 'manuscript_type'.

        """
        return self.inventory.filers[manuscript_type]

    """ Core siMpLify Methods """
    
    @abstract_method
    def draft(self) -> None:
        """Subclasses must provide their own methods."""
        pass

    @abstract_method
    def publish(self) -> None:
        """Subclasses must provide their own methods."""
        pass

    def apply(self, **kwargs) -> 'Manuscript':
        """Subclasses must provide their own methods."""
        pass
    
    
@dataclass
class Author(Creator):
    """Builds completed Book instances.

    Args:
        idea (Idea): an instance of Idea containing siMpLify project settings.
        inventory (Inventory): an instance of Inventory containing file and
            folder management attributes and methods.
        options (Optional[Union['ManuscriptOptions', Dict[str, 'Manuscript']]]):

        auto_publish (Optional[bool]): whether to call the 'publish' method when
            the class is instanced.

    """
    idea: 'Idea'
    inventory: 'Inventory'
    options: Optional[Union['ManuscriptOptions', Dict[str, 'Manuscript']]] = field(
        default_factory = dict)
    steps: Optional[Union[List[str], str]] = field(default_factory = list)
    auto_publish: Optional[bool] = True

    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _create_steps(self, name: str) -> List[str]:
        """Gets 'steps' from Idea, if not passed to class instance."""
        if not self.steps:
            try:
                return listify(self.idea[name]['_'.join([name, 'steps'])])
            except KeyError:
                pass
        return self
    
    def _create_techniques(self, 
            name: str, 
            steps: List[str]) -> List[List[str]]:
        """Tries to get techniques from shared Idea instance.
        
        Args:
            name (str): name of class to be built.
            steps (List[str]): steps in class to be built.

        Returns:
            List[List[str]] of parallel sequences of steps.

        """
        possibilities = []
        for step in steps:
            try:
                possibilities.append(listify(
                    self.idea[name]['_'.join([step, 'techniques'])]))
            except KeyError:
                possibilities.append(['none'])
        return list(map(list, product(*possibilities)))

    def _create_chapters(self,
            name: str,
            technique: str) -> List['Chapter']:
        """
        """
        chapters = []
        steps = self._create_steps(name = name)
        possibilities = self._create_techniques(name = name, steps = steps)
        for i, sequence in enumerate(possibilities):
            self.chapters_create.apply(
                number = i,
                steps = dict(zip(self.steps, sequence)))
        return chapters

    """ Core siMpLify Methods """

    def draft(self) -> None:
        return self

    def publish(self) -> None:
        """Creates instance of ChapterCreator."""
        self.chapter_creator = ChapterCreator(
            idea = self.idea,
            inventory = self.inventory,
            auto_publish = self.auto_publish)
        return self

    def apply(self,
            name: str,
            options: Optional['ManuscriptOptions'] = None) -> 'Book':
        """Creates a Manuscript object based upon arguments passed.

        Args:

        Returns:
            Manuscript instance.

        """
        return self.creators[manuscript_type].apply(
            name = name,
            manuscript_object = manuscript_object,
            technique = technique)
            
            name: str,
            book: Optional['Book'] = Book,
            technique: Optional[str] = None) -> 'Manuscript':
        """Creates a Manuscript object based upon arguments passed.

        Args:
            manuscript_type (str): either 'book', 'chapter', or 'page'.
            manuscript_object (Optional['Manuscript']): if the generic Book,
                Chapter, or Page is not to be used, an alternative class
                should be passed.

        Returns:
            Manuscript instance.

        """
        parameters = {}
        for need in self.needs[manuscript_type]:
            parameters[need] = getattr(self, '_'.join(['_draft', need]))()
        if manuscript_object is None:
            return self.default[manuscript_type](parameters)
        else:
            return manuscript_object(parameters)

       
@dataclass
class ChapterCreator(Creator):
    """Base class for building Manuscript classes and instances.

    Args:
        idea (Idea): an instance of Idea containing siMpLify project settings.
        inventory (Inventory): an instance of Inventory containing file and
            folder management attributes and methods.

    """
    idea: 'Idea'
    inventory: 'Inventory'
    options: Optional[Union['ManuscriptOptions', Dict[str, 'Manuscript']]] = field(
        default_factory = dict)
    auto_publish: Optional[bool] = True

    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    def draft(self) -> None:
        """Sets initial attributes."""
        self.needs = ['pages', 'filer']
        self.default = Chapter
        return self


@dataclass
class PageCreator(Creator):
    """Base class for building Manuscript classes and instances.

    Args:
        idea (Idea): an instance of Idea containing siMpLify project settings.
        inventory (Inventory): an instance of Inventory containing file and
            folder management attributes and methods.

    """
    idea: 'Idea'
    inventory: 'Inventory'
    auto_publish: Optional[bool] = True

    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    def draft(self) -> None:
        """Sets initial attributes."""
        self.needs = ['algorithm', 'parameters', 'filer']
        self.default = Page
        return self


@dataclass
class Manuscript(ABC):
    """Base class for data processing, analysis, and visualization.

    SimpleComposite implements a modified composite tree pattern for organizing
    the various subpackages in siMpLify.

    Args:
        options (Optional[Union['ManuscriptOptions', Dict[str, Any]]]): allows
            setting of 'options' property with an argument. Defaults to None.

    """
    options: (Optional[Union['ManuscriptOptions', Dict[str, Any]]]) = None

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns '_children' dictionary as iterable."""
        return iter(self._children)

    """ Private Methods """

    def _draft_options(self) -> None:
        """Subclasses should provide their own methods to create 'options'.

        If the subclasses also allow for passing of '_options', the code below
        should be included as well.Any

        """
        if self._options is None:
            self._options = ManuscriptOptions(options = {}, _author = self)
        elif isinstance(self._options, Dict):
            self._options = ManuscriptOptions(
                options = self._options,
                _author = self)
        return self

    def _draft_techniques(self) -> None:
        """If 'techniques' does not exist, gets 'techniques' from 'idea'.

        If there are no matching 'steps' or 'techniques' in 'idea', a list with
        'none' is created for 'techniques'.

        """
        self.compare = False
        if self.techniques is None:
            try:
                self.techniques = getattr(
                    self.idea, '_'.join([self.name, 'steps']))
            except AttributeError:
                try:
                    self.compare = True
                    self.techniques = getattr(
                        self.idea, '_'.join([self.name, 'techniques']))
                except AttributeError:
                    self.techniques = ['none']
        else:
            self.techniques = listify(self.techniques)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Required method that sets default values."""
        # Injects attributes from Idea instance, if values exist.
        self = self.idea.apply(instance = self)
        # initializes core attributes.
        self._draft_options()
        self._draft_techniques()
        return self

    def publish(self) -> None:
        """Required method which applies methods to passed data.

        Subclasses should provide their own 'publish' method.

        Args:
            data (Optional[object]): an optional object needed for the method.

        """
        if data is None:
            data = self.ingredients
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
        for technique in self.techniques:
            data = self.options[technique].options.apply(
                key = technique,
                data = data,
                **kwargs)
        return data

    """ Composite Methods and Properties """

    def add_children(self, keys: Union[List[str], str]) -> None:
        """Adds outline(s) to '_children' from 'options' based on key(s).
        Args:
            keys (Union[List[str], str]): key(s) to 'options'.
        """
        for key in listify(keys):
            self._children[key] = self.options[key]
        return self

    def proxify(self) -> None:
        """Creates proxy names for attributes and methods."""
        try:
            proxy_attributes = {}
            for name, proxy in self.proxies.items():
                for key, value in self.__dict__.items():
                    if name in key:
                        proxy_attributes[key.replace(name, proxy)] = value
            self.__dict__.update(proxy_attributes)
        except AttributeError:
            pass
        return self

    @property
    def parent(self) -> 'Manuscript':
        """Returns '_parent' attribute."""
        return self._parent

    @parent.setter
    def parent(self, parent: 'Manuscript') -> None:
        """Sets '_parent' attribute to 'parent' argument.
        Args:
            parent (Manuscript): Manuscript class up one level in
                the composite tree.
        """
        self._parent = parent
        return self

    @parent.deleter
    def parent(self) -> None:
        """Sets 'parent' to None."""
        self._parent = None
        return self

    @property
    def children(self) -> Dict[str, Union['Outline', 'Manuscript']]:
        """Returns '_children' attribute.
        Returns:
            Dict of str access keys and Outline or Manuscript values.
        """
        return self._children

    @children.setter
    def children(self, children: Dict[str, 'Outline']) -> None:
        """Assigns 'children' to '_children' attribute.
        If 'override' is False, 'children' are added to '_children'.
        Args:
            children (Dict[str, 'Outline']): dictionary with str for reference
                keys and values of 'Manuscript'.
        """
        self._children = children
        return self

    @children.deleter
    def children(self, children: Union[List[str], str]) -> None:
        """ Removes 'children' for '_children' attribute.
        Args:
            children (Union[List[str], str]): key(s) to children classes to
                remove from '_children'.
        """
        for child in listify(children):
            try:
                del self._children[child]
            except KeyError:
                pass
        return self

    """ Strategy Methods and Properties """

    def add_options(self,
            options: Union['ManuscriptOptions', Dict[str, Any]]) -> None:
        """Assigns 'options' to '_options' attribute.

        Args:
            options (options: Union['ManuscriptOptions', Dict[str, Any]]): either
                another 'ManuscriptOptions' instance or an options dict.

        """
        self.options += options
        return self

    @property
    def options(self) -> 'ManuscriptOptions':
        """Returns '_options' attribute."""
        return self._options

    @options.setter
    def options(self, options: Union['ManuscriptOptions', Dict[str, Any]]) -> None:
        """Assigns 'options' to '_options' attribute.

        Args:
            options (Union['ManuscriptOptions', Dict[str, Any]]): ManuscriptOptions
                instance or a dictionary to be stored within a ManuscriptOptions
                instance (this should follow the form outlined in the
                ManuscriptOptions documentation).

        """
        if isinstance(options, dict):
            self._options = ManuscriptOptions(options = options)
        else:
            self._options.add(options = options)
        return self

    @options.deleter
    def options(self, options: Union[List[str], str]) -> None:
        """ Removes 'options' for '_options' attribute.

        Args:
            options (Union[List[str], str]): key(s) to options classes to
                remove from '_options'.

        """
        for option in listify(options):
            try:
                del self._options[option]
            except KeyError:
                pass
        return self