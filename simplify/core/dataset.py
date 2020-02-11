"""
.. module:: dataset
:synopsis: data containment made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from ast import literal_eval
from collections.abc import Container
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd

from simplify.core.utilities import listify


@dataclass
class Dataset(object):
    """Collection of associated pandas data objects.

    Args:
        data (Optional[Union[pd.DataFrame, np.ndarray, str, Path]]): a dataset
            for all pandas objects to be derived or file path information for
            such an object to be imported. Defaults to None.
        datatypes (Optional[Dict[str, str]]): keys are column names and values
            are siMpLify proxy datatypes. Defaults to an empty dictionary.
        prefixes (Optional[Dict[str, str]]): keys are column prefixes and
            values are siMpLify proxy datatypes. Defaults to an empty
            dictionary.
        idea (ClassVar['Idea']): shared 'Idea' instance with project settings.
        inventory (ClassVar['Inventory']): shared 'Inventory' instance with
            project file management settings.

    """
    data: Optional[Union[pd.DataFrame, np.ndarray, str, Path]] = None
    datatypes: Optional[Dict[str, str]] = field(default_factory = dict)
    prefixes: Optional[Dict[str, str]] = field(default_factory = dict)
    idea: ClassVar['Idea'] = None
    inventory: ClassVar['Inventory'] = None

    def __post_init__(self) -> None:
        """Sets instance attributes."""
        self._set_folder_defaults()
        self._initialize_bunches()
        self.types = DataTypes()
        self._initialize_datatypes()
        self.stages = DataStages(parent = self)
        return self

    """ Factory and Validation Class Methods """

    @classmethod
    def create(cls,
            data: Optional[Union[pd.DataFrame, np.ndarray, str, Path]] = None,
            x: Optional[Union[pd.DataFrame, np.ndarray, str, Path]] = None,
            y: Optional[Union[pd.Series, np.ndarray, str, Path]] = None,
            datatypes: Optional[Dict[str, str]] = None,
            prefixes: Optional[Dict[str, str]] = None,
            idea: Optional['Idea'] = None,
            inventory: Optional['Inventory'] = None) -> 'Dataset':
        """Creates an Dataset instance.

        Either 'data' or 'x' and 'y' should be passed to Datatset, but not both.
        Both options are provided because datasets are sometimes supplied with
        'x' and 'y' already divided.

        Args:
            data (Optional[Union[pd.DataFrame, np.ndarray, str, Path]]): a
                dataset for all pandas objects to be derived or file path
                information for such an object to be imported. Defaults to None.
            x (Optional[Union[pd.DataFrame, np.ndarray, str, Path]]): a dataset
                with all features for data analysis or file path information for
                such an object to be imported. Defaults to None.
            y (Optional[Union[pd.Series, np.ndarray, str, Path]]): a
                1-dimensional data object containing the dependent variable or
                file path information for such an object to be imported.
                Defaults to None.
            datatypes (Optional[Dict[str, str]]): keys are column names and
                values are siMpLify proxy datatypes. Defaults to an empty
                dictionary.
            prefixes (Optional[Dict[str, str]]): keys are column prefixes and
                values are siMpLify proxy datatypes. Defaults to an empty
                dictionary.
            idea (Optional['Idea']): shared 'Idea' instance with project
                settings.
            inventory (Optional['Inventory']): shared 'Inventory' instance with
                project file management settings.

        Returns:
            Dataset instance, properly configured.

        Raises:
            TypeError: if 'data' is neither a file path, file folder,
                None, DataFrame, Series, numpy array, or Dataset instance.

        ToDo:
            Make 'x' and 'y' combination work for non-DataFrames

        """
        # Injects 'idea' and 'inventory' into 'Dataset' class, if passed.
        if idea is not None:
            cls.idea = idea
        if inventory is not None:
            cls.inventory = inventory
        if x is not None and y is not None and data is None:
            data = x
            data[cls.idea['analyst']['label']] = y
        # Creates 'Dataset' based upon argumnets passed.
        if isinstance(data, Dataset):
            return data
        elif data is None:
            return cls()
        elif isinstance(data, (pd.DataFrame, np.ndarray, str)):
            if datatypes is None:
                datatypes = dict()
            if prefixes is None:
                prefixes = dict()
            return cls(
                data = cls._validate_data(data = data),
                datatypes = datatypes,
                prefixes = prefixes)
        elif isinstance(data, pd.Series):
            # To do add row to DataFrame.
            pass
        else:
            raise TypeError(' '.join(
                ['data must be a file path, file folder, DataFrame, Series',
                'None, Dataset, dict, or numpy array']))

    @classmethod
    def _validate_data(cls,
            data: Union[pd.DataFrame, np.ndarray, str, Path]) -> pd.DataFrame:
        """Validates 'data' as or converts 'data' to a pandas DataFrame.

        Args:
            data (Union[pd.DataFrame, np.ndarray, str]): a pandas DataFrame,
                numpy array, or path (in string or Path form) of a file with
                data to be loaded into a pandas DataFrame.

        Returns:
            pd.DataFrame: derived from 'data'.

        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data = data)
        elif isinstance(data, (str, Path)):
            return cls.inventory.load(file_path = data)

    """ Dunder Methods """

    def __getattr__(self,
            attribute: str) -> Union['DataBunch', pd.DataFrame, pd.Series]:
        if attribute in ['train', 'training']:
            return self.__dict__[self.__dict__['train_set']]
        elif attribute in ['test', 'testing']:
            return self.__dict__[self.__dict__['test_set']]
        elif attribute in ['x']:
            return self.__dict__['full_bunch'].x
        elif attribute in ['y']:
            return self.__dict__['full_bunch'].y
        elif attribute in ['x_train']:
            return self.__dict__[self.__dict__['train_set']].x
        elif attribute in ['y_train']:
            return self.__dict__[self.__dict__['train_set']].y
        elif attribute in ['x_test']:
            return self.__dict__[self.__dict__['test_set']].x
        elif attribute in ['y_test']:
            return self.__dict__[self.__dict__['test_set']].y
        elif attribute in ['x_val']:
            return self.__dict__['val_bunch'].x
        elif attribute in ['y_val']:
            return self.__dict__['val_bunch'].y
        # Returns appropriate lists of columns with datatype 'attribute'.
        else:
            try:
                if attribute in self.__dict__['types'].groups:
                    return self._get_columns_by_type(datatype = attribute[:-1])
                # Combines 'floats' and 'integers' into 'numerics'.
                elif attribute in ['numerics']:
                    return self.floats + self.integers
            except KeyError:
                try:
                    return getattr(self.__dict__['data'], attribute)
                except (AttributeError, KeyError):
                    raise KeyError(' '.join(
                        [attribute, 'is not in', self.__class__.__name__]))

    def __setattr__(self,
            attribute: str,
            value: Union['DataBunch', pd.DataFrame, pd.Series]) -> None:
        if attribute in ['train', 'training']:
            self.__dict__[self.__dict__['train_set']] = value
        elif attribute in ['test', 'testing']:
            self.__dict__[self.__dict__['test_set']] = value
        elif attribute in ['x']:
            self.__dict__['full_bunch'].x = value
        elif attribute in ['y']:
            self.__dict__['full_bunch'].y = value
        elif attribute in ['x_train']:
            self.__dict__[self.__dict__['train_set']].x = value
        elif attribute in ['y_train']:
            self.__dict__[self.__dict__['train_set']].y = value
        elif attribute in ['x_test']:
            self.__dict__[self.__dict__['test_set']].x = value
        elif attribute in ['y_test']:
            self.__dict__[self.__dict__['test_set']].y = value
        elif attribute in ['x_val']:
            self.__dict__['val_bunch'].x = value
        elif attribute in ['y_val']:
            self.__dict__['val_bunch'].y = value
        else:
            self.__dict__[attribute] = value

    def __len__(self) -> int:
        """Returns length of 'data' instance.

        Returns:
            int: length of 'data'.

        """
        return len(self.data)

    def __iter__(self) -> Iterable:
        """Returns iterable of 'data'.

        Returns:
            Iterable: of pandas data object stored in 'data'.

        """
        return iter(self.data.iterrows())

    def __add__(self, other: Union[pd.DataFrame, pd.Series]) -> None:
        """Adds 'other' to stored data.

        Args:
            other (Union[pd.DataFrame, pd.Series]): data to add.

        """
        self.add(name = other.name, data = other)
        return self

    def __iadd__(self, other: Union[pd.DataFrame, pd.Series]) -> None:
        """Adds 'other' to stored data.

        Args:
            other (Union[pd.DataFrame, pd.Series]): data to add.

        """
        self.add(name = other.name, data = other)
        return self

    """ Private Methods """

    def _initialize_bunches(self) -> None:
        self._create_bunches()
        self._create_mapping()
        return self

    def _create_bunches(self) -> None:
        self.full_bunch = DataBunch(name = 'full')
        self.train_bunch = DataBunch(name = 'training')
        self.test_bunch = DataBunch(name = 'testing')
        self.val_bunch = DataBunch(name = 'validation')
        return self

    def _create_mapping(self) -> None:
        self.train_set = self.train_bunch
        self.test_set = self.test_bunch
        return self

    def _set_folder_defaults(self) -> None:
        self.import_folder = 'processed'
        self.export_folder = 'processed'
        return self

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
        return self

    """ Public Methods """

    def add(self, data: Union[pd.DataFrame, pd.Series]) -> None:
        """Adds 'data' to stored data.

        Args:
            data (Union[pd.DataFrame, pd.Series]): data to add.

        ToDo:
            Add flexible method for different merging and joining of pandas
                DataFrames.

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
            self.data[name] = self.types.convert(
                proxy_type = datatype,
                column = self.data[name])
            self.datatypes[name] = datatype
        return self

    def create_xy(self,
            data: Optional[pd.DataFrame] = None,
            label: Optional[str] = None) -> 'DataBunch':
        """Splits 'data' into 'x' and 'y' based upon the label passed.

        Args:
            data (Optional[pd.DataFrame]): a pandas DataFrame to be divided.
                Defaults to None. If passed, it replaces the local 'data'
                attribute.
            label (Optional[str]): name of column to be stored in 'y'. Defaults
                to None. If not passed, the method looks for 'label' in 'idea'.

        Returns:
            'DataBunch': with 'x' and 'y' initialized.

        """
        if data is not None:
            self.data = self._validate_data(data = data)
        if label is None:
            try:
                label = self.idea['analyst']['label']
            except KeyError:
                label = 'label'
        x_columns = list(self.data.columns.values)
        x_columns.remove(label)
        self.full_bunch.x = self.data[x_columns]
        self.full_bunch.y = self.data[label]
        if not hasattr(self, 'label_datatype'):
            self.label_datatype = self.datatypes[label]
            del self.datatypes[label]
        self.stages.change('full')
        return self

    def downcast(self, columns: Optional[Union[List[str], str]] = None) -> None:
        """Decreases memory usage by downcasting datatypes.

        If 'columns' is not passed, all columns are downcast.

        Args:
            columns (Optional[Union[List[str], str]]): columns to downcast.

        """
        for name in self._check_columns(columns):
            self.data[name] = self.types.downcast(
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
            row[column] = self.default_values[datatype]
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
            self.datatypes[name] = self.types.infer(column = self.data[name])
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
            self.datatypes.update({name: 'integer'})
            if assign_index:
                self.data.set_index(name, inplace = True)
        except (TypeError, AttributeError):
            raise TypeError('To add an index, data must be a pandas DataFrame')
        return self


@dataclass
class DataBunch(object):

    name: str
    x: Optional[pd.DataFrame] = None
    y: Optional[pd.Series] = None

    def __post_init__(self) -> None:
        if self.x is not None:
            self._start_columns = self.x.columns.values
        else:
            self._start_columns = []
        return self

    @property
    def dropped_columns(self) -> List[str]:
        if self._start_columns:
            return self._start_columns - self.x.columns.values
        else:
            return []


@dataclass
class DataTypes(Container):

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
            'full': DataStage(
                train_set = 'full_bunch',
                test_set = 'full_bunch'),
            'testing': DataStage(
                train_set = 'train_bunch',
                test_set = 'test_bunch'),
            'validating': DataStage(
                train_set = 'train_bunch',
                test_set = 'val_bunch')}
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
