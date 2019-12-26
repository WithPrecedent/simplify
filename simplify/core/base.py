"""
.. module:: base
:synopsis: siMpLify base classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import Container
from collections.abc import Iterator
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.utilities import listify
from simplify.core.utilities import pathlibify


@dataclass
class SimplePublisher(ABC):
    """Base class for creating Projects, Books, Chapters, and Pages.

    Args:
        project (Optional['Project')]: a related Project instance. Defaults to
            None. If no 'project' is passed, it is assigned to self.

    """
    project: Optional['Project'] = None

    def __post_init__(self) -> None:
        """ Sets initial attributes and calls 'draft' method."""
        if self.project is None:
            self.project = self
        self.draft()
        return self

    """ Private Methods """

    def _publish_idea(self,
            manuscript: 'SimpleManuscript',
            name: str) -> 'SimpleManuscript':
        """Drafts attributes from 'idea'.

        Args:
            manuscript (Union['Book', 'Chapter', 'Page']): siMpLify class
                instance to be modified.

        Returns:
            manuscript (Union['Book', 'Chapter', 'Page']): siMpLify class
                instance with modifications made.

        """
        sections = ['general', manuscript.name]
        try:
            sections.extend(listify(manuscript.idea_sections))
        except AttributeError:
            pass
        for section in sections:
            try:
                for key, value in self.project.idea[section].items():
                    if not hasattr(manuscript, key):
                        setattr(manuscript, key, value)
            except KeyError:
                pass
        return manuscript

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self,
            manuscript: Union[
                'Book', 'Chapter', 'Page']) -> Union['Book', 'Chapter', 'Page']:
        """Subclasses must provide their own methods."""
        pass

    @abstractmethod
    def publish(self,
            manuscript: Union[
                'Book', 'Chapter', 'Page']) -> Union['Book', 'Chapter', 'Page']:
        """Subclasses must provide their own methods."""
        pass


@dataclass
class SimpleManuscript(Collection, ABC):
    """Base class for Book, Chapter, and Page iterables."""

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        # Automatically calls 'draft' method.
        try:
            self.draft()
        except AttributeError:
            pass
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        # Calls 'apply' method if 'auto_apply' is True.
        if hasattr(self, 'auto_apply') and self.auto_apply:
            self.apply()
        return self

    """ Required ABC Methods """

    def __contains__(self, item: str) -> bool:
        """Returns whether 'attribute' exists in the class iterable.

        Args:
            item (str): name of item to check.

        Returns:
            bool: whether the 'item' exists in the class iterable.

        """
        return item in getattr(self, self.iterable)

    def __iter__(self) -> Iterable:
        """Returns class iterable."""
        return iter(getattr(self, self.iterable))

    def __len__(self) -> int:
        """Returns length of class iterable."""
        return len(getattr(self, self.iterable))


@dataclass
class SimpleWorker(Iterator):
    """Applies methods to siMpLify class instances.

    Args:
        project ('Project'): a related director class instance.

    """
    project: 'Project'
    sequence: 'MutableSequence'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets initial index location for iterable.
        self._position = 0
        return self

    """ Required ABC Methods """

    def __next__(self) -> Any:
        """Returns current item in 'sequence' at '_position'.

        Returns:
            Any: item in 'sequence' at '_position'.

        """
        try:
            self._position += 1
            return self.sequence[self._position - 1]
        except IndexError:
            raise StopIteration()

    """ Private Methods """

    def _apply_gpu(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients', 'SimpleManuscript']] = None,
            **kwargs) -> NotImplementedError:
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's 'apply' method.

        Raises:
            NotImplementedError: until dynamic GPU support is added.

        """
        raise NotImplementedError(
            'GPU support outside of modeling is not yet supported')

    def _apply_multi_core(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients',
                'SimpleManuscript']] = None) -> 'SimpleManuscript':
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        with Pool() as pool:
            pool.imap(manuscript.apply, data)
        pool.close()
        return self

    def _apply_single_core(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients', 'SimpleManuscript']] = None,
            **kwargs) -> 'SimpleManuscript':
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's 'apply' method.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        manuscript.apply(data = data, **kwargs)
        return self

    """ Core siMpLify Methods """

    def apply(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients', 'SimpleManuscript']] = None,
            **kwargs) -> 'SimpleManuscript':
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's options' 'apply' method.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        if self.parallelize and not kwargs:
            self._apply_multi_core(
                manuscript = manuscript,
                data = data)
        else:
            self._apply_single_core(
                manuscript = manuscript,
                data = data,
                **kwargs)
        return manuscript


@dataclass
class SimpleSettings(MutableMapping):
    """Base class for mono-state siMpLify dictionaries.

    Args:
        project ('Project'): associated Project instance.
        configuration (Optional[Dict[str, Any]]): dictionary storing class
            instance dictionary to which access keys point. Defaults to an
            empty dictionary.

    """
    project: 'Project'
    configuration: Optional[Dict[str, Any]] = field(default_factory = dict)

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in 'configuration'.

        Args:
            item (str): name of key in 'configuration'.

        """
        try:
            del self.configuration[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> Any:
        """Returns item in 'configuration'.

        Args:
            item (str): name of key in 'configuration'.

        Returns:
            Any: item stored as a 'configuration' value.

        Raises:
            KeyError: if 'item' is not found in 'configuration' and does
                not match a recognized wildcard.

        """
        try:
            return self.configuration[item]
        except KeyError:
            raise KeyError(' '.join(
                [item, 'is not in', self.manuscript.name, 'configuration']))

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets 'item' in 'configuration' to 'value'.

        Args:
            item (str): name of key in 'configuration'.
            value (Any): value to be paired with 'item' in 'configuration'.

        """
        self.configuration[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'configuration'."""
        return iter(self.configuration)

    def __len__(self) -> int:
        """Returns length of 'configuration'."""
        return len(self.configuration)

    """ Other Dunder Methods """

    def __add__(self, other: Union['SimpleSettings', Dict[str, Any]]) -> None:
        """Combines two SimpleOptions instances or 'configuration' dictionaries.

        Args:
            other (Union['SimpleSettings', Dict[str, Any]]): either another
                'SimpleSettings' instance or a 'configuration' dictionary.

        """
        self.add(settings = other)
        return self

    def __iadd__(self, other: Union['SimpleSettings', Dict[str, Any]]) -> None:
        """Combines two SimpleOptions instances or 'configuration' dictionaries.

        Args:
            other (Union['SimpleSettings', Dict[str, Any]]): either another
                'SimpleSettings' instance or a 'configuration' dictionary.

        """
        self.add(settings = other)
        return self

    """ Public Methods """

    def add(self,
            key: Optional[str] = None,
            value: Optional[Any] = None,
            settings: Optional[Union[
                'SimpleSettings', Dict[str, Any]]] = None) -> None:
        """Adds arguments to 'configuration' dictionary.

        Args:
            key (Optional[str]): dictionary key for 'value' to use. Defaults to
                None.
            value (Optional[Any]): dictionary value to be associated with 'key'.
                Defaults to None.
            settings (Optional[Union['SimpleSettings', Dict[str, Any]]]):
                either another 'SimpleSettings' instance or dictionary. Defaults
                to None.

        """
        if key is not None and value is not None:
            self.configuration[key] = value
        if settings is not None:
            try:
                self.configuration.update(settings.configuration)
            except AttributeError:
                try:
                    self.configuration.update(settings)
                except (TypeError, AttributeError):
                    pass
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Subclasses may provide their own methods."""
        return self

    def publish(self) -> None:
        """Subclasses may provide their own methods."""
        return self


@dataclass
class SimpleOptions(MutableMapping):
    """Base class for variable-state siMpLify dictionaries.

    SimpleOptions subclasses support variable-state mutable mappings. By
    default, the 'active' dictionary is 'options' which means that access
    methods look to the 'options' dictionary until the '_change_active' method
    is called. By calling '_change_active', a new dictionary is automatically
    created and access methods point toward it. If a subclass wants to use a
    different name than 'options' for the internal dictionary, it should set
    'active' to that name before calling super().__post_init__(). Unless
    overridden, 'active' changes to 'published' when the 'publish' method is
    called and 'applied' when the 'apply' method is called. But outside the
    class, access methods will always point to the 'active' dictionary with
    no references made to the specific dictionary that is active. Another
    attribute 'previous_active' is stored once the active state is changed to
    indicate the previous active dictionary.

    All values in the internal dictionary must be Resource-compatible. This
    allows the lazy-loader ('load' method) to be called by a SimpleOptions
    instance.

    Args:
        project ('Project'): associated Project instance.
        options (Optional[Dict[str, 'Resource']]): SimpleOptions instance or
            a SimpleOptions-compatible dictionary. Defaults to an empty
            dictionary.
        steps (Optional[Union[List[str], str]]): steps of key(s) to iterate in
            'options'. Also, if not reset by the user, 'steps' is used if the
            'default' property is accessed. Defaults to an empty list.

    """
    options: Optional[Dict[str, 'Resource']] = field(default_factory = dict)
    steps: Optional[Union[List[str], str]] = field(default_factory = list)

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets initial '_default' to 'steps'.
        self._default = self.steps
        # Initializes state management for accessing active stored dictionary.
        if not hasattr(self, 'active'):
            self.active = 'options'
        self.active = SimpleState(
            states = [self.active, 'published', 'applied'],
            initial_state = self.active)
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in the active dictionary.

        Args:
            item (str): name of key in the active dictionary.

        """
        try:
            del getattr(self, self.active)[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> Any:
        """Returns item in the active dictionary.

        If there are no matches, the method searches for a matching wildcard.

        Args:
            item (str): name of key in the active dictionary.

        Returns:
            Any: item stored as a the active dictionary value.

        Raises:
            KeyError: if 'item' is not found in the active dictionary and does
                not match a recognized wildcard.

        """
        try:
            return getattr(self, self.active)[item]
        except KeyError:
            try:
                return self.wildcards[item]
            except (KeyError, AttributeError):
                raise KeyError(' '.join(
                    [item, 'is not in', self.manuscript.name, 'options']))

    def __setitem__(self, item: str, value: 'Resource') -> None:
        """Sets 'item' in the active dictionary to 'value'.

        Args:
            item (str): name of key in the active dictionary.
            value ('Resource'): value to be paired with 'item' in the
                active dictionary.

        """
        getattr(self, self.active)[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of the active dictionary if 'iterable' not set."""
        try:
            return iter(getattr(self, self.iterable))
        except AttributeError:
            return iter(getattr(self, self.active))

    def __len__(self) -> int:
        """Returns length of the active dictionary if 'iterable' not set.."""
        try:
            return len(getattr(self, self.iterable))
        except AttributeError:
            return len(getattr(self, self.active))

    """ Other Dunder Methods """

    def __add__(self,
            other: Union['SimpleOptions', Dict[str, 'Resource']]) -> None:
        """Combines arguments with the active dictionary.

        Args:
            other (Union['SimpleOptions', Dict[str, 'Resource']]): another
                'SimpleOptions' instance or compatible dictionary.

        """
        self.add(options = other)
        return self

    def __iadd__(self,
            other: Union['SimpleOptions', Dict[str, 'Resource']]) -> None:
        """Combines arguments with the active dictionary.

        Args:
            other (Union['SimpleOptions', Dict[str, 'Resource']]): another
                'SimpleOptions' instance or compatible dictionary.

        """
        self.add(options = other)
        return self

    """ Private Methods """

    def _change_active(self,
            new_active: str,
            copy_previous: Optional[bool] = False) -> None:
        """Changes 'active' for access methods.

        Args:
            new_active (str): name of new state and dictionary to be created.
            copy_previous (Optional[bool]): whether to copy the old dictionary
                into the new state dictionary. If False, an empty dictionary is
                created for the new state. Defaults to False.

        """
        self.active.change(new_state = new_active)
        if not hasattr(self, self.active):
            if copy_previous:
                setattr(self, self.active, getattr(
                    self, self.active.previous).copy())
            else:
                setattr(self, self.active, {})
        return self

    def _draft_wildcards(self):
        """Sets wildcard values to check if a key doesn't exist."""
        self.wildcards = {
            'all': self.all,
            'default': self.default,
            'defaults': self.default,
            'none': ['none']}
        return self

    """ Public Methods """

    def add(self,
            key: Optional[str] = None,
            value: Optional['Resource'] = None,
            options: Optional[Union[
                'SimpleOptions', Dict[str, 'Resource']]] = None) -> None:
        """Combines arguments with the active dictionary.

        Args:
            key (Optional[str]): dictionary key for 'value' to use. Defaults to
                None.
            value (Optional['Resource']): siMpLify object to store in the
                active dictionary dictionary. Defaults to None.
            options (Optional[Union['SimpleOptions',
                Dict[str, 'Resource']]]): another 'SimpleOptions' instance
                or a compatible dictionary. Defaults to None.

        """
        if key is not None and value is not None:
            getattr(self, self.active)[key] = value
        if options is not None:
            try:
                getattr(self, self.active).update(getattr(options, self.active))
            except AttributeError:
                try:
                    getattr(self, self.active).update(options)
                except (TypeError, AttributeError):
                    pass
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        # Sets wildcard options.
        self._draft_wildcards()
        return self

    def publish(self, options: Union[List[str], Dict[str, Any], str]) -> None:
        """Loads, creates, and finalizes instances in the active dictionary.

        Args:
            options (Union[List[str], Dict[str, Any], str]): list of keys,
                dictionary, or a string indicating which options should be
                loaded and instanced. If a dictionary is passed, its keys will
                be used to find matching stored 'options'.

        """
        self._change_active(new_active = 'published')
        if isinstance(options, dict):
            options = list(options.keys())
        for option in listify(options):
            # Lazily loads all selected Resource instances.
            loaded = getattr(self, self.previous_active)[option].load()
            instance = loaded()
            instance.publish()
            getattr(self, self.active)[option] = instance
        return self


    """ Wildcard Properties """

    @property
    def all(self) -> List[str]:
        """Returns list of keys of the active dictionary.

        Returns:
            List[str] of keys stored in the active dictionary.

        """
        return list(self.keys())

    @property
    def default(self) -> None:
        """Returns '_default' or list of keys of the active dictionary.

        Returns:
            List[str] of keys stored in '_default' or the active dictionary.

        """
        if self._default:
            return self._default
        else:
            return self.all

    @default.setter
    def default(self, options: Union[List[str], str]) -> None:
        """Sets '_default' to 'options'

        Args:
            'options' (Union[List[str], str]): list of keys in the active
                dictionary to return when 'default' is accessed.

        """
        self._default = listify(options)
        return self

    @default.deleter
    def default(self, options: Union[List[str], str]) -> None:
        """Removes 'options' from '_default'.

        Args:
            'options' (Union[List[str], str]): list of keys in the active
                dictionary to remove from '_default'.

        """
        for option in listify(options):
            try:
                del self._default[option]
            except KeyError:
                pass
        return self


@dataclass
class Resource(Container):
    """Object construction instructions used by SimpleOptions.

    Ideally, this class should have no additional methods beyond the lazy
    loader ('load' method) and __contains__ dunder method.

    Users can use the idiom 'x in Option' to check if a particular attribute
    exists and is not None. This means default values for optional arguments
    should generally be set to None to allow use of that idiom.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify module).
        component (str): name of python object within 'module' to load (can
            either be a siMpLify or non-siMpLify object).

    """
    name: str
    module: str
    component: str

    """ Required ABC Methods """

    def __contains__(self, attribute: str) -> bool:
        """Returns whether attribute exists in a subclass instance.

        Args:
            attribute (str): name of attribute to check.

        Returns:
            bool: whether the attribute exists and is not None.

        """
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    """ Public Methods """

    def load(self) -> object:
        """Returns object from module based upon instance attributes.

        Returns:
            object from module indicated in passed Option instance.

        """
        return getattr(import_module(self.module), self.component)


@dataclass
class SimpleSequence(MutableSequence):
    """Base class for ordered sequences.

    SimpleSequence subclasses should NOT store integers. Doing so conflicts
    with the __getitem__ and __delitem__ methods which allow for index and str
    searches of stored lists.

    """
    sequence: Optional[Union[List[str], str]] = field(default_factory = list)

    def __post_init__(self) -> None:
        """Ensures 'sequence' is a list'."""
        self.sequence = listify(self.sequence, default_empty = True)
        return self

    """ Dunder Methods """

    def __getitem__(self, item: Union[int, str, slice]) -> Union[str, int]:
        """Returns value at index or returns index at value.

        Args:
            item (Union[int, str, slice]): index, str, or slice in 'sequence'.

        Returns:
            int if str passed, str if int passed, or list if slice passed.

        """
        if isinstance(index, slice):
                return [self[i] for i in range(*item.indices(len(self)))]
        else:
            try:
                return self.sequence[item]
            except TypeError:
                return self.sequence.index(item)

    def __setitem__(self, index: int, value: str) -> None:
        """Sets item in 'sequence' at 'index' to 'value'.

        Args:
            index (int): index in 'sequence' to add 'value'.
            value (str): value to add to 'sequence' at 'index'.

        """
        self.sequence[index] = value
        return self

    def __delitem__(self, item: Union[int, str]) -> None:
        """Deletes 'item' whether an index number or string in 'sequence'.

        Args:
            item (Union[int, str]): either index or str in list.

        """
        try:
            del self.sequence[item]
        except TypeError:
            self.sequence.remove(item)
        return self

    def __iter__(self) -> Iterable:
        """Returns 'sequence' as an Iterable."""
        return iter(self.sequence)

    def __len__(self) -> int:
        """Returns length of 'sequence'."""
        return len(self.sequence)

    """ Other Dunder Methods """

    def __repr__(self) -> List:
        """Returns 'sequence' as a list."""
        return self.__str__()

    def __str__(self) -> List:
        """Returns 'sequence' as a list."""
        return listify(self.sequence)

    """ Public Methods """

    def add(self, items: Union[List[str], str]) -> None:
        """Adds items to stored sequence at the end.

        Args:
            items (Union[List[str], str]): item(s) to add to stored sequence.

        """
        if (isinstance(items, MutableSequence)
                or issubclass(items, MutableSequence)):
            self.sequence.extend(items)
        else:
            self.sequence.append(items)
        return self

    def insert(self, index: int, value: str) -> None:
        """Applies 'insert' method to 'sequence'.

        Args:
            index (int): index in 'sequence' list to insert 'value'.
            value (str): value to insert at 'index' in 'sequence'.

        """
        self.sequence.insert(index, value)
        return self

    def next(self, item: Union[int, str]) -> str:
        """Returns next item in 'sequence' after 'item'.

        Args:
            item (Union[int, str]): either index or str in list.

        Returns:
            str of next item in 'sequence'.

        """
        if isinstance(item, int):
            return self.sequence[item + 1]
        else:
            return self[self[item] + 1]


@dataclass
class SimpleState(Container):
    """Base class for state management."""

    states: List[str]
    initial_state: Optional[str] = None

    def _post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Required ABC Methods """

    def __contains__(self, attribute: str) -> bool:
        """Returns whether 'attribute' exists in 'states'.

        Args:
            attribute (str): name of state to check.

        Returns:
            bool: whether the attribute exists in 'states'.

        """
        return attribute in self.states

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        """Returns string name of 'state'."""
        return self.publish()

    def __str__(self) -> str:
        """Returns string name of 'state'."""
        return self.publish()

    """ State Management Methods """

    def change(self, new_state: str) -> None:
        """Changes 'state' to 'new_state'.

        Args:
            new_state(str): name of new state matching a string in 'states'.

        Raises:
            TypeError: if new_state is not in 'states'.

        """
        if new_state in self.states:
            self.previous = self.state
            self.state = new_state
        else:
            raise TypeError(' '.join([new_state, 'is not a recognized state']))

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates state machine default settings. """
        if self.initial_state:
            self.state = self.initial_state
        else:
            self.state = self.states[0]
        self.previous = self.state
        return self

    def publish(self) -> None:
        """Returns string name of 'state'."""
        return self.state


@dataclass
class SimpleDistributor(ABC):
    """Base class for siMpLify Importer and Exporter."""

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        self.draft()
        if self.auto_publish:
            self.publish()
        return self

    """ Private Methods """

    def _check_kwargs(self,
            variables_to_check: List[str],
            passed_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Selects kwargs for particular methods.

        If a needed argument was not passed, default values are used.

        Args:
            variables_to_check (List[str]): variables to check for values.
            passed_kwargs (Dict[str, Any]): kwargs passed to method.

        Returns:
            new_kwargs (Dict[str, Any]): kwargs with only relevant parameters.

        """
        new_kwargs = passed_kwargs
        for variable in variables_to_check:
            if not variable in passed_kwargs:
                if variable in self.default_kwargs:
                    new_kwargs.update(
                        {variable: self.inventory.default_kwargs[variable]})
                elif hasattr(self, variable):
                    new_kwargs.update({variable: getattr(self, variable)})
        return new_kwargs

    """ Core siMpLify Methods """

    def draft(self) -> None:
        self.options = SimpleOptions(options = {
            'csv': 'csv',
            'matplotlib': 'mp',
            'pandas': 'pd',
            'pickle': 'pickle'})
        return self


@dataclass
class SimplePath(MutableMapping):
    """Base class for variable-state folder or file paths.

    Args:
        inventory ('Inventory): related Inventory instance.
        folder (str): folder where 'names' are or should be.
        names (Dict[str, str]): dictionary where keys are names of states and
            values are Path objects linked to those states.

    """
    inventory: 'Inventory'
    folder: str
    names: Dict[str, str]

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in 'names'.

        Args:
            item (str): name of key in 'names'.

        """
        try:
            del self.names[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> Path:
        """Returns item in 'names'.

        Args:
            item (str): name of key in 'names'.

        Returns:
            Path: value stored as a 'names'.

        Raises:
            KeyError: if 'item' is not found in 'names'.

        """
        try:
            return self.names[item]
        except KeyError:
            raise KeyError(' '.join([item, 'is not in Inventory']))

    def __setitem__(self, item: str, value: Path) -> None:
        """Sets 'item' in 'names' to 'value'.

        Args:
            item (str): name of key in 'names'.
            value (Path): value to be paired with 'item' in 'names'.

        """
        self.names[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'names'."""
        return iter(self.names)

    def __len__(self) -> int:
        """Returns length of 'names'."""
        return len(self.names)

    """ Other Dunder Methods """

    def __repr__(self) -> Path:
        """Returns value from 'names' based upon current 'state'."""
        return self.publish()

    def __str__(self) -> Path:
        """Returns value from 'names' based upon current 'state'."""
        return self.publish()

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Converts values in 'names' from str to Path objects."""
        new_names = {}
        for state, name in self.names.items():
            new_names[state] = pathlibify(path = folder.joinpath(name))
            if new_names[state].is_dir():
                self.inventory.create_folder(path = new_names[state])
        self.names = new_names
        return self

    def publish(self) -> Path:
        """Returns value from 'names' based upon current 'state'."""
        return self.names[self.inventory.state]