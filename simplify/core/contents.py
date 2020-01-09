"""
.. module:: options
:synopsis: siMpLify base lexicon classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.utilities import listify


@dataclass
class Contents(MutableMapping, ABC):
    """Base class for storing options and strategies."""

    def __post_init__(self) -> None:
        """Initializes core attributes."""
        # Sets name of internal 'lexicon' dictionary.
        if not hasattr(self, 'lexicon'):
            self.lexicon = 'options'
        # Sets name of 'wilcards' which correspond to properties.
        if not hasattr(self, 'wildcards'):
            self.wildcards = ['default', 'all']
        return self

    """ Required ABC Methods """

    def __delitem__(self, key: str) -> None:
        """Deletes item in the 'lexicon' dictionary.

        Args:
            key (str): name of key in the 'lexicon' dictionary.

        """
        try:
            del getattr(self, self.lexicon)[key]
        except KeyError:
            pass
        return self

    def __getitem__(self, key: str) -> Any:
        """Returns item in the 'lexicon' dictionary.

        If there are no matches, the method searches for a matching wildcard in
        attributes.

        Args:
            key (str): name of key in the 'lexicon' dictionary.

        Returns:
            Any: item stored as a the 'lexicon' dictionary value.

        Raises:
            KeyError: if 'key' is not found in the 'lexicon' dictionary.

        """
        try:
            return getattr(self, self.lexicon)[key]
        except KeyError:
            if item in self.wildcards:
                return getattr(self, key)
            else:
                raise KeyError(' '.join(
                    [key, 'is not in', self.__class__.__name__]))

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in the 'lexicon' dictionary to 'value'.

        Args:
            key (str): name of key in the 'lexicon' dictionary.
            value (Any): value to be paired with 'key' in the 'lexicon'
                dictionary.

        """
        getattr(self, self.lexicon)[key] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of the 'lexicon' dictionary.

        Returns:
            Iterable stored in the 'lexicon' dictionary.

        """
        return iter(getattr(self, self.lexicon))

    def __len__(self) -> int:
        """Returns length of the 'lexicon' dictionary if 'iterable' not set..

        Returns:
            Integer of length of 'lexicon' dictionary.

        """
        return len(getattr(self, self.lexicon))

    """ Other Dunder Methods """

    def __add__(self, other: Union['Contents', Dict[str, Any]]) -> None:
        """Combines argument with the 'lexicon' dictionary.

        Args:
            other (Union['Contents', Dict[str, Any]]): another
                'Contents' instance or compatible dictionary.

        """
        self.add(options = other)
        return self

    def __iadd__(self, other: Union['Contents', Dict[str, Any]]) -> None:
        """Combines argument with the 'lexicon' dictionary.

        Args:
            other (Union['Contents', Dict[str, Any]]): another
                'Contents' instance or compatible dictionary.

        """
        self.add(options = other)
        return self

    """ Public Methods """

    def add(self,
            key: Optional[str] = None,
            value: Optional[Any] = None,
            options: Optional[Union[
                'Contents', Dict[str, Any]]] = None) -> None:
        """Combines arguments with the 'lexicon' dictionary.

        Args:
            key (Optional[str]): dictionary key for 'value' to use. Defaults to
                None.
            value (Optional[Any]): item to store in the 'lexicon' dictionary.
                Defaults to None.
            options (Optional[Union['Contents', Dict[str, Any]]]):
                another 'Contents' instance or a compatible dictionary.
                Defaults to None.

        """
        if key is not None and value is not None:
            getattr(self, self.lexicon)[key] = value
        if options is not None:
            try:
                getattr(self, self.lexicon).update(
                    getattr(options, options.lexicon))
            except AttributeError:
                try:
                    getattr(self, self.lexicon).update(options)
                except (TypeError, AttributeError):
                    pass
        return self

    """ Wildcard Properties """

    @property
    def all(self) -> List[str]:
        """Returns list of keys of the 'lexicon' dictionary.

        Returns:
            List[str] of keys stored in the 'lexicon' dictionary.

        """
        return list(self.keys())

    @property
    def default(self) -> List[str]:
        """Returns '_default' or list of keys of the 'lexicon' dictionary.

        Returns:
            List[str] of keys stored in '_default' or the 'lexicon' dictionary.

        """
        try:
            return self._default
        except AttributeError:
            self._default = self.all
            return self._default

    @default.setter
    def default(self, options: Union[List[str], str]) -> None:
        """Sets '_default' to 'options'

        Args:
            'options' (Union[List[str], str]): list of keys in the lexicon
                dictionary to return when 'default' is accessed.

        """
        self._default = listify(options)
        return self

    @default.deleter
    def default(self, options: Union[List[str], str]) -> None:
        """Removes 'options' from '_default'.

        Args:
            'options' (Union[List[str], str]): list of keys in the lexicon
                dictionary to remove from '_default'.

        """
        for option in listify(options):
            try:
                del self._default[option]
            except KeyError:
                pass
            except AttributeError:
                self._default = self.all
                del self.default[options]
        return self


@dataclass
class SimpleOptions():
    """Base class for storing options and strategies.

    Args:
        project (Optional['Project']): associated Project instance. Defaults to
            None.
        options (Optional[str, Any]): default stored dictionary. Defaults to an
            empty dictionary.
        null_value (Optional[Any]): value to return when 'none' is accessed.
            Defaults to None.

    """
    project: Optional['Project'] = None
    options: Optional[Dict[str, Any]] = field(default_factory = dict)
    null_value: Optional[Any] = None

    def __post_init__(self) -> None:
        """Sets name of internal 'lexicon' dictionary."""
        self.lexicon = 'options'
        self.wildcards = ['default', 'all', 'none']
        super().__post_init__()
        return self

    """ Wildcard Properties """

    @property
    def none(self) -> None:
        """Returns 'null_value'.

        Returns:
            'null_value' attribute or None.

        """
        return self.null_value

    @default.setter
    def none(self, null_value: Any) -> None:
        """Sets 'none' to 'null_value'.

        Args:
            null_value (Any): value to return when 'none' is sought.

        """
        self.null_value = null_value
        return self

@dataclass
class SimpleData(Contents):
    """Base class for storing pandas data objects."""

    datasets: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]] = field(
        default_factory = dict)
    default_dataset: Optional[str] = 'df'

    def _post_init__(self) -> None:
        """Initializes class instance attributes."""
        if not hasattr(self, 'lexicon'):
            self.lexicon = 'datasets'
        self.wildcards = ['default', 'all', 'train', 'test', 'val']
        super().__post_init__()
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Dunder Methods """

    def __add__(self, other: Union['Contents', Dict[str, Any]]) -> None:
        """Combines argument with the 'lexicon' dictionary.

        Args:
            other (Union['Contents', Dict[str, Any]]): another
                'Contents' instance or compatible dictionary.

        """
        self.add(options = other)
        return self

    def __iadd__(self, other: Union['Contents', Dict[str, Any]]) -> None:
        """Combines argument with the 'lexicon' dictionary.

        Args:
            other (Union['Contents', Dict[str, Any]]): another
                'Contents' instance or compatible dictionary.

        """
        self.add(options = other)
        return self


    def __contains__(self, attribute: str) -> bool:
        """Returns whether 'attribute' is in 'datasets' or matches a wildcard.

        Args:
            attribute (str): name of dataset to check.

        Returns:
            bool: whether the attribute exists in 'datasets'.

        """
        return ((attribute in self.datasets and self.datasets[attribute])
                or attribute in self.wildcards)

    """ Public Methods """

    def add(self,
            key: Optional[str] = None,
            value: Optional[Any] = None,
            options: Optional[Union[
                'Contents', Dict[str, Any]]] = None) -> None:
        """Combines arguments with the 'lexicon' dictionary.

        Args:
            key (Optional[str]): dictionary key for 'value' to use. Defaults to
                None.
            value (Optional[Any]): item to store in the 'lexicon' dictionary.
                Defaults to None.
            options (Optional[Union['Contents', Dict[str, Any]]]):
                another 'Contents' instance or a compatible dictionary.
                Defaults to None.

        """
        if key is not None and value is not None:
            getattr(self, self.lexicon)[key] = value
        if options is not None:
            try:
                getattr(self, self.lexicon).update(
                    getattr(options, options.lexicon))
            except AttributeError:
                try:
                    getattr(self, self.lexicon).update(options)
                except (TypeError, AttributeError):
                    pass
        return self



@dataclass
class SimpleDatatypes(Contents):

    self.datatypes = {
        'boolean': bool,
        'float': float,
        'integer': int,
        'string': object,
        'categorical': CategoricalDtype,
        'list': list,
        'datetime': datetime64,
        'timedelta': timedelta}