"""
.. module:: options
:synopsis: base class for storing different options
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.utilities import listify
from simplify.core.utilities import proxify


@dataclass
class Options(MutableMapping):
    """Base class for different options to be stored.

    The Options class should be injected with the shared Idea instance before it
    or any subclass is instanced. This is done automatically through the normal
    siMpLify Project access point. But if you are creating a highly customized
    workflow, this step must be taken for siMpLify to work properly.

    Args:
        options (Optional[Union[Dict[str, Any], 'Options']]): a dictionary with
            various strategies or a completed Options instance.
        defaults (Optional[Union[List[str], str]]): key(s) to use if the
            'default' property is selected. Defaults to an empty list. If not
            specified, and 'default' options are sought, all options will be
            returned.
        parent (Optional[object]): related class instance.
        name (Optional[str]): name of options for error messages. If 'name' is
            not provided, __class__.__name__.lower() is used instead.

    """
    options: Optional[Union[Dict[str, Any], 'Options']] = field(
        default_factory = dict)
    defaults: Optional[Union[List[str], str]] = field(default_factory = list)
    parent: Optional[object] = None
    name: Optional[str] = None

    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        # Sets private 'parent' attribute.
        self._parent = parent
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name') or self.name is not None:
            self.name = self.__class__.__name__.lower()
        # Validates 'options' argument.
        self._check_options()
        # Sets default 'state' to allow mono and variable state options.
        self.state = 'options'
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in options.

        Args:
            item (str): name of key in options.

        """
        try:
            del getattr(self, self.state)[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> Any:
        """Returns item in options.

        If there are no matches, the method searches for a matching wildcard.

        Args:
            item (str): name of key in options.

        Returns:
            Any: item stored as an 'options' value.

        Raises:
            KeyError: if 'item' is not found in options and does not match
                a recognized wildcard.

        """
        try:
            return getattr(self, self.state)[item]
        except KeyError:
            try:
                return self.wildcards[item]
            except KeyError:
                raise KeyError(' '.join([item, 'is not in', self.name]))

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets 'item' in options to 'value'.

        Args:
            item (str): name of key in options.
            value (Any): value to be paired with 'item' in options.

        """
        getattr(self, self.state)[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of options."""
        return iter(getattr(self, self.state))

    def __len__(self) -> int:
        """Returns length of options."""
        return len(getattr(self, self.state))

    """ Other Dunder Methods """

    def __add__(self, other: Union[Dict[str, Any], 'Options']) -> None:
        """Combines two options dictionaries.

        Args:
            other (Union[Dict[str, Any], 'Options'): either another 'Options'
                instance or an options dict.

        Raises:
            TypeError: if 'other' is neither a 'Options' instance nor a dict.

        """
        self.add(options = other)
        return self

    def __iadd__(self, other: Union[Dict[str, Any], 'Options']) -> None:
        """Combines two options dictionaries.

        Args:
            other (Union[Dict[str, Any], 'Options'): either another 'Options'
                instance or an options dict.

        Raises:
            TypeError: if 'other' is neither a 'Options' instance nor a dict.

        """
        self.add(options = other)
        return self

    def __invert__(self) -> None:
        """Reverses keys and values in options."""
        setattr(self, self.state, self.__reversed__())
        return self

    def __reversed__(self) -> Dict[Any, str]:
        """Returns options with keys and values reversed.

        Returns:
            Dict (Any, str): dictionary with keys and values reversed.Any
        """
        return {value: key for key, value in getattr(self, self.state).items()}

    """ Private Methods """

    def _check_options(self):
        """Validates type of passed 'options' argument.

        Raises:
            TypeError: if 'options' is neither a dictionary nor Options instance
                or subclass.

        """
        if (isinstance(self.options, Options)
                or issubclass(self.options, Options)):
            self = self.options
        elif not isinstance(self.options, Dict):
            raise TypeError('options must be a dict or Options type')
        return self

    """ Public Methods """

    def add(self, options: Union[Dict[str, Any], 'Options']) -> None:
        """Combines two options dictionaries.

        Args:
            options (Union[Dict[str, Any], 'Options'): either another 'Options'
                instance or an options dict.

        Raises:
            TypeError: if 'other' is neither a 'Options' instance nor a dict.

        """
        try:
            getattr(self, self.state).update(getattr(other, self.state))
        except AttributeError:
            try:
                getattr(self, self.state).update(other)
            except AttributeError:
                raise TypeError(' '.join(
                    ['addition requires objects to be dict or Options types']))
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        # Applies proxy attribute names, if any are set.
        try:
            self = proxify(instance = self, proxies = self.proxies)
        except AttributeError:
            pass
        # Sets wildcard values to check if a key doesn't exist in options.
        self.wildcards = {
            'all': self.all,
            'default': self.default,
            'defaults': self.default,
            'none': ['none']}
        return self

    """ Relational Properties """

    @property
    def parent(self) -> object:
        """Returns related class instance.

        Returns:
            object stored in '_parent'.

        """
        return self._parent

    @parent.setter
    def parent(self, parent: object) -> None:
        """Sets related class instance.

        Args:
            parent (object): related class instance.

        """
        self._parent = parent
        return self

    @parent.deleter
    def parent(self) -> None:
        """Changes '_parent' to None."""
        self._parent = None
        return self

    """ Wildcard Properties """

    @property
    def all(self) -> List[str]:
        """Returns list of keys of current 'state' dictionary.

        Returns:
            list (str) of keys stored in activate state of 'options'.

        """
        return list(self.keys())

    @property
    def default(self) -> None:
        """Returns 'defaults' or list of keys of current 'state'
        dictionary.

        Returns:
            list (str) of keys stored in 'defaults' or in the activate
                state of 'options'.

        """
        return self.defaults or self.all

    @default.setter
    def default(self, options: Union[List[str], str]) -> None:
        """Sets 'defaults' to 'options'.

        Args:
            options (Union[List[str], str]): list of keys in 'options' to return
                when 'default' is passed.

        """
        self.defaults = listify(options)
        return self

    @default.deleter
    def default(self, options: Union[List[str], str]) -> None:
        """Removes 'options' from 'defaults'.

        Args:
            options (Union[List[str], str]): list of keys in 'options' to remove
                from 'defaults'.

        """
        for option in listify(options):
            try:
                del self.defaults[option]
            except KeyError:
                pass
        return self


@dataclass
class ManuscriptOptions(Options):
    """Base class for Manuscript subclasses with 'options'.

    Args:
        options (Optional[Dict[str, Any]]): alternative strategies stored
            in a dictionary in the following format:

                {str: Outline}

            If subclassing, 'drafted' should be declared in the 'draft' method.
            Defaults to an empty dict.
        defaults (Optional[Union[List[str], str]]): key(s) to use if the
            'default' property is selected. Defaults to an empty list. If not
            specified, and 'default' options are sought, all options will be
            returned.
        parent (Optional[object]): related class instance.
        name (Optional[str]): name of options for error messages. If 'name' is
            not provided, __class__.__name__.lower() is used instead.

    """
    options: Optional[Dict[str, Any]] = field(default_factory = dict)
    defaults: Optional[Union[List[str], str]] = field(default_factory = list)
    parent: Optional[object] = None
    name: Optional[str] = None

    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        self.proxies = {'parent': 'manuscript'}
        self.state = 'drafted'
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        super().draft()
        self.state = 'drafted'
        self.arguments = {
        }
        return self

    def publish(self, steps: Dict[str, str]) -> None:
        """Loads, creates, and finalizes instances in 'options'.

        Args:
            steps (Dict[str, str]): dictionary with keys of step name and values
                of technique names. If the step names are the same as the
                technique names, a 'technique' argument is not passed when
                creating the instance of the selected class.

        """
        # Sets state for access methods.
        self.state = 'published'
        # Instances and publishes all selected options.
        for step, technique in steps.items():
            # Lazily loads all stored options from stored Outline instances.
            option = self.drafted[step].load()
            if step == technique:
                instance = option()
            else:
                instance = option(technique = technique)
            instance.publish()
            self.published[step] = instance
        return self

    def apply(self,
            technique: str,
            data: Optional[object] = None,
            **kwargs) -> object:
        """Calls 'apply' method for published option matching 'technique'.

        Args:
            technique (str): technique for specific option to be applied.
            data (Optional[object]): object for option to be applied. Defaults
                to None.
            kwargs: any additional parameters to pass to the option's 'apply'
                method.

        Returns:
            object is returned if data is passed, otherwise None is returned.

        """
        # Sets state for access methods.
        self.state = 'applied'
        data = self.published[technique].apply(data = data)
        self.applied[technique] = self.published[technique]
        return data