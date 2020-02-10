"""
.. module:: dataset
:synopsis: data containment made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from ast import literal_eval
from collections.abc import Container
from collections.abc import Mapping
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd

from simplify.core.utilities import listify


@dataclass
class Dataset(MutableMapping):
    """Contains a collection of data objects (DataSlice instances).

    Args:
        dataset (Optional[Union['Dataset', 'DataSlice', pd.DataFrame,
            np.ndarray, str, Dict[str, Union['DataSlice', pd.DataFrame,
            np.ndarray, str]]]]): keys are names of the data objects (e.g. 'x',
            'y_train', etc.) and values are DataSlice instances storing pandas
            data objects. Defaults to an empty dictionary.
        idea (ClassVar['Idea']): shared 'Idea' instance with project settings.
        inventory (ClassVar['Inventory']): shared 'Inventory' instance with
            project file management settings.

    """
    data: Optional[Union[
        'Dataset',
        'DataSlice',
        pd.DataFrame,
        np.ndarray,
        str,
        Dict[str, Union[
            'DataSlice',
            pd.DataFrame,
            np.ndarray,
            str]]]] = None
    idea: ClassVar['Idea'] = None
    inventory: ClassVar['Inventory'] = None

    def __post_init__(self) -> None:
        """Sets default attributes."""
        self.stages = DataStages(parent = self)
        self.types = PandasTypes()
        self._create_dataslices()
        self._set_defaults()
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            data: Union[
                'Dataset',
                pd.DataFrame,
                pd.Series,
                np.ndarray,
                str,
                Dict[str, Union[
                    'DataSlice', pd.DataFrame, pd.Series, np.ndarray, str]]],
            idea: Optional['Idea'] = None,
            inventory: Optional['Inventory'] = None) -> 'Dataset':
        """Creates an Dataset instance.

        Args:
            data (Union['Dataset', pd.DataFrame, pd.Series, np.ndarray, str,
                Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray, str]]]):
                Dataset instance or information needed to create one.
            inventory ('Inventory'): a Inventory instance.

        Returns:
            Dataset instance, properly configured.

        Raises:
            TypeError: if 'data' is neither a file path, file folder,
                None, DataFrame, Series, numpy array, or Dataset instance.

        """
        if idea is not None:
            cls.idea = idea
        if inventory is not None:
            cls.inventory = inventory
        if isinstance(data, Dataset):
            return data
        elif data is None:
            return cls()
        elif isinstance(data,
                (list, dict, pd.Series, pd.DataFrame, np.ndarray, str)):
            new_data = cls()
            if isinstance(data, dict):
                for name, data in data.items():
                    new_data.add(name = name, data = data)
            elif isinstance(data,
                    (pd.Series, pd.DataFrame, np.ndarray, str)):
                new_data.add(name = 'full', data = data)
            return new_data
        else:
            raise TypeError(' '.join(
                ['data must be a file path, file folder, DataFrame, Series',
                'None, Dataset, dict, or numpy array']))

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Any:
        """Returns value for 'key' in '__dict__'.

        If there are no matches, the method searches for a matching wildcard
        option ('all', 'train', 'test', 'val', or 'xy').

        Args:
            key (str): name of key in '__dict__'.

        Returns:
            Any: item stored in '__dict__' or a wildcard value.

        """
        try:
            return self.__dict__[key]
        except KeyError:
            try:
                return self.full[key]
            except (TypeError, KeyError):
                if key in ['train', 'train_set', 'training']:
                    return (self.__dict__[self.train_set[0]],
                        self.__dict__[self.train_set[1]])
                elif key in ['test', 'test_set', 'testing']:
                    return (self.__dict__[self.test_set[0]],
                        self.__dict__[self.test_set[1]])
                else:
                    raise KeyError(' '.join(
                        [key, 'is not in', self.__class__.__name__]))

    def __setitem__(self, key: str, value = 'DataSlice') -> None:
        """Sets 'key' to 'value'.

        Args:
            key (str): key for 'DataSlice' instance to be stored.
            value ('DataSlice'): instance to be stored.

        """
        self.__dict__[key] = value

    def __delitem__(self, key: str) -> Any:
        """Deletes stored 'DataSlice' instance at 'key'.

        Args:
            key (str): key to 'DataSlice' to delete.

        """
        del self.__dict__[key]

    def __len__(self) -> int:
        """Returns length of 'full' 'DataSlice' instance.

        Returns:
            int: length of 'full' 'DataSlice'.

        """
        return len(self.full)

    def __iter__(self) -> Iterable:
        """Returns iterable of 'full' 'DataSlice'.

        Returns:
            Iterable: of 'DataSlice' object currently set to 'full'.

        """
        return iter(self.full)

    """ Other Dunder Methods """

    def __getattr__(self, attribute: str) -> Any:
        """Tries to find 'attribute' in 'full' 'DataSlice' instance.

        Args:
            attribute (str): attribute to look for in 'full' 'DataSlice'
                instance.

        Returns:
            Any: attribute in 'full' 'DataSlice' instance.

        Raises:
            AttributeError: if 'attribute' is not found in 'full' 'DataSlice'
                instance.

        """
        try:
            if attribute in ['train', 'train_set', 'training']:
                return (self.__dict__[self.__dict__['train_set'][0]],
                    self.__dict__[self.__dict__['train_set'][1]])
            elif attribute in ['test', 'test_set', 'testing']:
                return (self.__dict__[self.__dict__['test_set'][0]],
                    self.__dict__[self.__dict__['test_set'][0]])
        except (AttributeError, KeyError):
            return getattr(self.__dict__['full'], attribute)

    def __setattr__(self, attribute: str, value: Any) -> None:
        """Sets attribute in DataSet instance or in 'full' 'DataSlice'.

        If 'attribute' is listed in '_slices', it will be added to the 'Dataset'
        instance. If not, it will be added to the 'full' 'DataSlice'.

        Args:
            attribute (str): name of attribute to be set.
            value (Any): value of attribute to be set.

        """
        try:
            if attribute in self._slices:
                self.__dict__[attribute] = value
            else:
                setattr(self.__dict__['full'], attribute, value)
        except (KeyError, AttributeError, TypeError):
            self.__dict__[attribute] = value

    def __add__(self, other: 'DataSlice') -> None:
        """Adds 'other' to stored data.

        Args:
            other ('DataSlice'): an 'DataSlice' instance.

        """
        self.add(name = other.name, data = other)
        return self

    def __iadd__(self, other: 'DataSlice') -> None:
        """Adds 'other' to stored data.

        Args:
            other ('DataSlice'): an 'DataSlice' instance.

        """
        self.add(name = other.name, data = other)
        return self

    def __repr__(self) -> str:
        return self.__dict__.__repr__()

    def __str__(self) -> str:
        return self.__dict__.__str__()

    """ Private Methods """

    def _create_dataslices(self) -> None:
        self._slices = [
            'full',
            'x', 'y',
            'x_train', 'y_train',
            'x_test', 'y_test',
            'x_val', 'y_val']
        for slice in self._slices:
            self.__dict__[slice] = DataSlice(dataset = self)
        if isinstance(self.data, pd.DataFrame):
            self.full = DataSlice(data = self.data, dataset = self)
        del self.data
        return self

    def _set_defaults(self) -> None:
        self.train_set = ('x_train', 'y_train')
        self.test_set = ('x_test', 'y_test')
        self.import_folder = 'processed'
        self.export_folder = 'processed'
        return self

    """ Public Methods """

    def add(self, name: str, data: 'DataSlice') -> None:
        """Adds 'data' to stored data.

        Args:
            data ('DataSlice'): an 'DataSlice' instance.

        """
        self.__dict__[name] = DataSlice.create(
            data = data,
            inventory = self.inventory,
            dataset = self)
        return self

    def divide_xy(self,
            data: Optional['DataSlice'] = None,
            label: Optional[str] = None) -> None:
        """Splits data into 'x' and 'y' based upon the label passed.

        Args:
            data (Optional['DataSlice']): instance storing a pandas DataFrame.
            label (Optional[str]): name of column to be stored in 'y'.

        """
        if data is not None:
            self.full = data
        if label is None:
            try:
                label = self.idea['analyst']['label']
            except KeyError:
                label = 'label'
        x_columns = list(self.full.columns.values)
        x_columns.remove(label)
        self.x = self.full[x_columns]
        self.y = self.full[label]
        self.label_datatype = self.full.datatypes[label]
        del self.full.datatypes[label]
        self.stages.change('divided')
        return self


@dataclass
class DataSlice(MutableMapping):
    """A container of a pandas data object with related metadata.

    Args:
        data (Union[pd.DataFrame, pd.Series]): a stored pandas data object.
        datatypes (Optional[Dict[str, str]]): keys are column names and values
            are siMpLify proxy datatypes. Defaults to an empty dictionary.
        prefixes (Optional[Dict[str, str]]): keys are column prefixes and
            values are siMpLify proxy datatypes. Defaults to an empty
            dictionary.
       dataset (Optional['Dataset']): related 'Dataset' instance.
            Defaults to None.
        name (Optional[str]): this should match the key used in a related
            'Dataset' instance, if one exists. This is used when any of the
            'add' methods is used to add this DataSlice instance to an
            'Dataset' instance. Defaults to 'data'.

    """
    data: Optional[Union[pd.DataFrame, pd.Series]] = None
    datatypes: Optional[Dict[str, str]] = field(default_factory = dict)
    prefixes: Optional[Dict[str, str]] = field(default_factory = dict)
    dataset: Optional['Dataset'] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        """Initializes instance attributes."""
        if self.data is not None:
            self._initialize_datatypes()
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            data: Union['DataSlice', pd.DataFrame, pd.Series, np.ndarray, str],
            inventory: Optional['Inventory'] = None,
            dataset: Optional['Dataset'] = None) -> 'DataSlice':
        """Creates an Dataset instance.

        Args:
            data (Union['DataSlice', pd.DataFrame, pd.Series, np.ndarray,
                str]): DataSlice instance or information needed to create one.
            inventory ('Inventory'): a Inventory instance.

        Returns:
            DataSlice instance, properly configured.

        Raises:
            TypeError: if 'data' is neither a file path, file folder,
                None, DataFrame, Series, numpy array, or DataSlice instance.

        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return cls(data = data, dataset = dataset)
        elif isinstance(data, np.ndarrray):
            return cls(data = pd.DataFrame(data = data), dataset = dataset)
        elif isinstance(data, str):
            if inventory is None:
                raise ValueError('inventory needed to load a data object')
            else:
                try:
                    return cls(
                        data = inventory.load(
                            folder = inventory.data,
                            file_name = data),
                        dataset = dataset)
                except FileNotFoundError:
                    try:
                        return cls(
                            data = inventory.load(file_path = data),
                            dataset = dataset)
                    except FileNotFoundError:
                        try:
                            inventory.create_batch(folder = data)
                            return cls(dataset = dataset)
                        except FileNotFoundError:
                            raise FileNotFoundError('data not found')
        else:
            raise TypeError(' '.join(
                ['data must be a file path, file folder, DataFrame, Series',
                'None, or numpy array']))

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Any:
        """Returns 'key' from 'data' attribute.

        Args:
            key (str): name of column in 'data' to return.

        Returns:
            pd.Series: column from 'data'.

        """
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        del self.data[key]
        return self

    def __iter__(self) -> Iterable:
        """Returns 'iterrows' method for 'data' attribute."""
        return self.data.iterrows()

    def __len__(self) -> int:
        """Returns integer length of 'data' attribute."""
        return len(self.data)

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __str__(self) -> str:
        return self.data.__str__()

    def __getattr__(self, attribute: str) -> Any:
        """Returns column lists or state-dependent variables.

        Args:
            attribute (str): attribute to look for.

        Returns:
            Any: attribute in state or List[str] of columns.

        Raises:
            AttributeError: if 'attribute' is not found in any of the searched
                attributes.

        """
        # Returns appropriate lists of columns with datatype 'attribute'.
        if attribute in self.dataset.types.groups:
            return self._get_columns_by_type(datatype = attribute[:-1])
        # Combines 'floats' and 'integers' into 'numerics'.
        elif attribute in ['numerics']:
            return self.floats + self.integers
        # Returns list of 'dropped_columns' by comparing to '_start_columns'.
        elif attribute in ['dropped_columns']:
            return self._start_columns - list(self.datatypes.keys())
        else:
            # Tries to return attribute from 'data' object itself.
            try:
                return getattr(self.data, attribute)
            except:
                raise AttributeError(' '.join(
                    [attribute, 'is not in', self.__class__.__name__]))

    def __add__(self,
            other: Union[pd.DataFrame, pd.Series, 'DataSlice']) -> None:
        """Combines 'other' with 'data'.

        Args:
            other (Union[pd.DataFrame, pd.Series, 'DataSlice']): a pandas data
                object or another 'DataSlice' instance.

        """
        self.add(data = other)
        return self

    def __iadd__(self,
            other: Union[pd.DataFrame, pd.Series, 'DataSlice']) -> None:
        """Combines 'other' with 'data'.

        Args:
            other (Union[pd.DataFrame, pd.Series, 'DataSlice']): a pandas data
                object or another 'DataSlice' instance.

        """
        self.add(data = other)
        return self

    """ Private Methods """

    def _check_columns(self, columns: Union[List[str], str] = None) -> List[str]:
        """Returns 'columns' keys if columns doesn't exist.

        Args:
            columns (Union[List[str], str]): column names.

        Returns:
            if columns is not None, returns columns, otherwise, the keys of
                the 'columns' attribute are returned.

        """
        if columns:
            return listify(columns)
        else:
            return list(self.data.columns.values)

    def _crosscheck_columns(self) -> None:
        """Harmonizes 'datatypes' dictionary with 'data' columns attribute."""
        for column in list(self.data.columns.values()):
            if column not in self.datatypes:
                self.datatypes[column] = self._infer_type(column = column)
        for column in list(self.datatypes.keys()):
            if column not in self.data.columns:
                del self.datatypes[column]
        return self

    def _get_columns_by_type(self, datatype: str) -> List[str]:
        """Returns list of columns of the specified datatype.

        Args:
            datatype (str): string matching datatype in 'datatypes'.

        Returns:
            list of columns matching the passed 'datatype'.

        """
        return [
            key for key, value in self.datatypes.items() if value == datatype]

    def _get_indices(self, columns: Union[List[str], str]) -> List[bool]:
        """Gets column indices for a list of column names.

        Args:
            columns (Union[List[str], str]): name(s) of columns for which
                indices are sought.

        Returns:
            bool mask for columns matching 'columns'.

        """
        return [
            self.data.columns.get_loc(column) for column in listify(columns)]

    def _initialize_datatypes(self) -> None:
        """Initializes datatypes for stored pandas data object."""
        if not self.datatypes:
            self.infer_datatypes()
        else:
            self._crosscheck_columns()
        self._start_columns = list(self.data.columns.values)
        return self

    """ Public Methods """

    def add(self,
            other: Union[pd.DataFrame, pd.Series, 'DataSlice']) -> None:
        """Combines 'other' with 'data'.

        Args:
            data (Union[pd.DataFrame, pd.Series, 'DataSlice']): a pandas
                data object or another 'DataSlice' instance.

        ToDo:
            Add functionality for all cases.

        """
        return self

    def change_datatype(self,
            columns: [Union[List[str], str]],
            datatype: str) -> None:
        """Changes column datatypes of 'columns'.

        The 'datatype' becomes the new datatype for the columns in both the
        'datatypes' dict and in reality - a method is called to try to convert
        the column to the appropriate datatype.

        Args:
            columns ([Union[List[str], str]]): column name(s) for datatypes to
                be changed.
            datatype (str): contains name of the datatype to convert the
                columns.

        """
        for name in listify(columns):
            self.data[name] = self.dataset.types.convert(
                proxy_type = datatype,
                column = self.data[name])
            self.datatypes[name] = datatype
        return self

    def downcast(self, columns: Optional[Union[List[str], str]] = None) -> None:
        """Decreases memory usage by downcasting datatypes.

        If 'columns' is not passed, all columns are downcast.

        Args:
            columns (Optional[Union[List[str], str]]): columns to downcast.

        """
        for name in self._check_columns(columns):
            self.data[name] = self.dataset.types.downcast(
                proxy_type = self.datatypes[name],
                column = self.data[name])
        return self

    def drop_columns(self,
            columns: Optional[Union[List[str], str]] = None) -> None:
        """Drops specified 'columns'.

        Args:
            columns (Optional[Union[List[str], str]]): columns to drop.

        """
        try:
            self.data.drop(columns, axis = 'columns', inplace = True)
        except TypeError:
            self.data.drop(columns, inplace = True)
        return self

    def get_series(self,
            columns: Optional[Union[List[str], str]] = None) -> None:
        """Creates a Series (row) with the 'datatypes' dict.

        Default values are added to each item in the series so that pandas does
        not automatically infer the datatype when a value is passed.

        Args:
            columns (Optional[Union[List[str], str]]): index names for pandas
                Series.

        Returns:
            pd.Series with index names matching 'columns' keys and datatypes
                matching 'datatypes' values.

        """
        row = pd.Series(index = self._check_columns(columns = columns))
        # Fills series with default_values based on datatype.
        for column, datatype in self.datatypes.items():
            row[column] = self.dataset.default_values[datatype]
        return row

    def infer_datatypes(self,
            columns: Optional[Union[List[str], str]] = None) -> None:
        """Infers proxy datatypes for 'columns'.

        If 'columns' is not passed, all columns are checked.

        Args:
            columns (Optional[Union[List[str], str]]): columns to infer the
                datatype for.

        """
        for name in self._check_columns(columns):
            self.datatypes[name] = self.dataset.types.infer(
                column = self.data[name])
        return self

    def uniquify(self,
            name: Optional[str] = 'index_universal',
            assign_index: Optional[bool] = False) -> None:
        """Creates a unique integer index for each row.

        Args:
            name (Optional[str]): column name for the index. Defaults to
                'index_universal'.
            assign_index (Optional[bool]): indicates whether the index
                'column' should be made the actual index of the DataFrame.
                Defaults to False.

        Raises:
            TypeError: if 'data' is not a DataFrame (usually because it is a
                pandas Series).

        """
        try:
            self.data[name] = range(1, len(self.data.index) + 1)
            self.datatypes.update({name, 'integer'})
            if assign_index:
                self.data.set_index(name, inplace = True)
        except (TypeError, AttributeError):
            raise TypeError('To add an index, data must be a pandas DataFrame')
        return self


@dataclass
class PandasTypes(Container):

    def __post_init__(self) -> None:
        self._create_proxies()
        self.options = list(self.proxies.keys())
        self.groups = [x + 's' for x in self.options]
        self._create_inferables()
        self._create_defaults()
        return self

    """ Required ABC Methods """

    def __contains__(self, item: str) -> bool:
        return item in self.options

    """ Private Methods """

    def _create_proxies(self) -> None:
        self._numpy_types = [
            'float16', 'float32', 'float64', 'floating', 'complexfloating',
            'complex64', 'complex128', 'int8', 'int16', 'int32', 'int64',
            'uint8', 'uint16', 'uint32', 'uint64', 'integer', 'signedinteger',
            'unsignedinteger', 'longlong', 'ulonglong', 'bool', 'object_',
            'datetime64', 'timedelta64']
        self._pandas_types = [
            'BooleanDtype', 'Int64Dtype', 'StringDtype', 'CategoricalDtype',
            'DatetimeTZDtype', 'IntervalDtype']
        self.proxies = {
            'float': ['float16', 'float32', 'float64', 'floating',
                'complexfloating', 'complex64', 'complex128'],
            'integer': ['int8', 'int16', 'int32', 'int64', 'Int64Dtype',
                'uint8', 'uint16', 'uint32', 'uint64', 'integer',
                'signedinteger', 'unsignedinteger', 'longlong', 'ulonglong'],
            'boolean': ['BooleanDtype', bool, 'bool'],
            'string': ['StringDtype', object, 'object_', 'object'],
            'categorical': ['CategoricalDtype'],
            'list': [list],
            'datetime': ['DatetimeTZDtype', 'datetime64', datetime],
            'timedelta': ['IntervalDtype', 'timedelta64', timedelta]}
        return self

    def _create_inferables(self) -> None:
        self.inferables = {}
        for proxy_type, group in self.proxies.items():
            for raw_type in group:
                self.inferables[raw_type] = proxy_type
        return self

    def _create_defaults(self) -> None:
        self.defaults = {}
        return self

    """ Public Methods """

    def convert(self,
            proxy_type: str,
            column: pd.Series,
            raise_errors: Optional[bool] = False) -> pd.Series:
        return self.downcast(
            proxy_type = proxy_type,
            column = column,
            raise_errors = raise_errors)

    def downcast(self,
            proxy_type: str,
            column: pd.Series,
            raise_errors: Optional[bool] = False) -> pd.Series:
        if raise_errors:
            raise_errors = 'raise'
        else:
            raise_errors = 'ignore'
        if proxy_type in ['list']:
            return column.apply(lambda x: literal_eval(str(x)))
        else:
            for raw_type in self.proxies[proxy_type]:
                if raw_type in self._numpy_types:
                    raw_type = getattr(np, raw_type)
                elif raw_type in self._pandas_types:
                    raw_type = getattr(pd, raw_type)
                try:
                    column.astype(raw_type, copy = False, errors = raise_errors)
                    break
                except (ValueError, TypeError):
                    pass
            return column

    def infer(self, column: pd.Series) -> str:
        try:
            return self.inferables[column.dtype]
        except KeyError:
            return self.inferables[str(column.dtype)]

    def infer_and_downcast(self, column: pd.Series) -> Tuple[str, pd.Series]:
        proxy_type = self.infer(column = column)
        return (
            proxy_type,
            self.downcast(proxy_type = proxy_type, column = column))


@dataclass
class DataStages(object):
    """Base class for data state management."""

    parent: object
    stages: Optional[Union[List[str], Dict[str, 'SimpleStage']]] = field(
        default_factory = dict)
    initial: Optional[str] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self._create_stages()
        self._set_current()
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            stages: Optional[Union[
                'SimpleStages',
                List[str],
                Dict[str, 'SimpleStage']]] = None) -> 'SimpleStages':
        """

        """
        if isinstance(stages, SimpleStages):
            return stages
        elif isinstance(stages, (list, dict)):
            return cls(stages = stages)
        elif stages is None:
            return cls()
        else:
            raise TypeError('stages must be a SimpleStages, dict, or list')

    """ Dunder Methods """

    def __repr__(self) -> str:
        """Returns string name of 'current'."""
        return self.current

    def __str__(self) -> str:
        """Returns string name of 'current'."""
        return self.current

    """ Private Methods """

    def _create_stages(self) -> None:
        self.stages = {
            'raw': DataStage(
                import_folder = 'raw',
                export_folder = 'raw'),
            'interim': DataStage(
                import_folder = 'raw',
                export_folder = 'interim'),
            'processed': DataStage(
                import_folder = 'interim'),
            'divided': DataStage(
                train_set = ('x', 'y'),
                test_set = ('x', 'y')),
            'testing': DataStage(
                train_set = ('x_train', 'y_train'),
                test_set = ('x_test', 'y_test')),
            'validating': DataStage(
                train_set = ('x_train', 'y_train'),
                test_set = ('x_val', 'y_val'))}
        return self

    def _set_current(self) -> None:
        """Sets current 'stage' upon initialization."""
        if self.initial and self.initial in self.stages:
            self.current = self.initial
        elif self.stages:
            self.current = list(self.stages.keys())[0]
        else:
            self.current = None
        self.previous = self.current
        return self

    """ Public Methods """

    def change(self, new_stage: str) -> None:
        """Changes 'stage' to 'new_stage'.

        Args:
            new_stage(str): name of new stage matching a string in 'stages'.

        Raises:
            TypeError: if new_stage is not in 'stages'.

        """
        if new_stage in self.stages:
            self.previous = self.current
            self.current = new_stage
            self.stages[self.current].apply(instance = self.parent)
        else:
            raise ValueError(' '.join([new_stage, 'is not a recognized stage']))

    """ Core siMpLify Methods """

    def apply(self) -> None:
        """Injects attributes into 'parent' based upon 'current'."""
        self.stages[self.current].apply(instance = self.parent)
        return self


@dataclass
class DataStage(object):
    """A single stage in data processing for a siMpLify project.

    Args:
        train_set (Optional[Tuple[str, str]]): names of attributes in a
            'Dataset' instance to return when 'train' is accessed. Defaults to
            None.
        test_set (Optional[Tuple[str, str]]): names of attributes in a
            'Dataset' instance to return when 'test' is accessed. Defaults to
            None.
        import_folder (Optional[str]): name of an attribute in an 'Inventory'
            instance corresponding to a path where data should be imported
            from. Defaults to 'processed'.
        export_folder (Optional[str]): name of an attribute in an 'Inventory'
            instance corresponding to a path where data should be exported
            from. Defaults to 'processed'.

    """
    train_set: Optional[Tuple[str, str]] = None
    test_set: Optional[Tuple[str, str]] = None
    import_folder: Optional[str] = field(default_factory = lambda: 'processed')
    export_folder: Optional[str] = field(default_factory = lambda: 'processed')

    """ Core SiMpLify Methods """

    def apply(self, instance: object) -> object:

        for attribute in [
                'train_set',
                'test_set',
                'import_folder',
                'export_folder']:
            setattr(instance, attribute, getattr(self, attribute))
        return instance
