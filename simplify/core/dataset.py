"""
.. module:: dataset
:synopsis: data containment made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from ast import literal_eval
from collections.abc import Container
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
        contents (Optional[Dict[str, 'DataSlice']]): keys are names of the data
            objects (e.g. 'x', 'y_train', etc.) and values are DataSlice
            instances storing pandas data objects. Defaults to an empty
            dictionary.
        default (Optional[str]): name of data object in 'contents' to apply
            methods to by default, if a particular data is not accessed.
            Defaults to 'data'.
        name (Optional[str]): this should be used to distinguish multiple sets
            of unrelated data. Ordinarily, it is not needed. Defaults to
            'contents'.
        idea ('Idea'): the shared 'Idea' instance with project settings.
        inventory ('Inventory):

    """
    idea: ClassVar['Idea'] = None
    inventory: ClassVar['Inventory'] = None
    dataset: Optional[Union[
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

    def __post_init__(self) -> None:
        """Sets default attributes."""
        self.active = 'full'
        self.types = PandasTypes()
        self._create_dataslices()
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            dataset: Union[
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
            dataset (Union['Dataset', pd.DataFrame, pd.Series, np.ndarray, str,
                Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray, str]]]):
                Dataset instance or information needed to create one.
            inventory ('Inventory'): a Inventory instance.

        Returns:
            Dataset instance, properly configured.

        Raises:
            TypeError: if 'dataset' is neither a file path, file folder,
                None, DataFrame, Series, numpy array, or Dataset instance.

        """
        if idea is not None:
            cls.idea = idea
        if inventory is not None:
            cls.inventory = inventory
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset,
                (list, dict, pd.Series, pd.DataFrame, np.ndarray, str)):
            new_dataset = cls()
            if dataset is None:
                pass
            elif isinstance(dataset, dict):
                for name, data in dataset.items():
                    new_dataset.add(name = name, data = data)
            elif isinstance(dataset,
                    (pd.Series, pd.DataFrame, np.ndarray, str)):
                new_dataset.add(name = 'full', data = dataset)
            return new_dataset
        else:
            raise TypeError(' '.join(
                ['dataset must be a file path, file folder, DataFrame, Series',
                'None, Dataset, dict, or numpy array']))

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Any:
        """Returns value for 'key' in 'contents'.

        If there are no matches, the method searches for a matching wildcard
        option.

        Args:
            key (str): name of key in 'contents'.

        Returns:
            Any: item stored in 'contents' or a wildcard value.

        """
        try:
            return self.__dict__[key]
        except KeyError:
            try:
                return self.__dict__[self.active][key]
            except (TypeError, KeyError):
                if key in ['all']:
                    return list(subsetify(self.__dict__, self._slices).values())
                elif key in ['train']:
                    keys = list(self.__dict__.keys())
                    training = [x for x in keys if x.endswith('train')]
                    return list(subsetify(self.__dict__, training).values())
                elif key in ['test']:
                    keys = list(self.__dict__.keys())
                    testing = [x for x in keys if x.endswith('test')]
                    return list(subsetify(self.__dict__, testing).values())
                elif key in ['val']:
                    keys = list(self.__dict__.keys())
                    validation = [x for x in keys if x.endswith('val')]
                    return list(subsetify(self.__dict__, validation).values())
                elif key in ['xy']:
                    return [self.__dict__['x'], self.__dict__['y']]
                else:
                    raise KeyError(' '.join(
                        [key, 'is not in', self.__class__.__name__]))

    def __setitem__(self, key: str, value = 'DataSlice') -> None:
        """
        """
        self.__dict__[key] = value


    def __delitem__(self, key: str) -> Any:
        """

        """
        del self.__dict__[key]

    def __len__(self) -> int:
        return len(getattr(self, self.active))

    def __iter__(self) -> Iterable:
        return len(getattr(self, self.active))

    """ Other Dunder Methods """

    def __getattr__(self, attribute: str) -> Any:
        """Tries to find attribute inside other attributes.

        Args:
            attribute (str): attribute to look for in other attributes.

        Returns:
            Any: attribute inside specific attributes.

        Raises:
            AttributeError: if 'attribute' is not found in any of the searched
                attributes.

        """
        if ('slices' in self.__dict__
                and attribute.lower() in self.__dict__['slices']):
            return self.__dict__[attribute.lower()]
        else:
            try:
                return getattr(self.__dict__['full'], attribute)
            except AttributeError:
                pass

    # def __setattr__(self, attribute: str, value: 'DataSlice') -> None:
    #     """Tries to find attribute inside other attributes.

    #     Args:
    #         attribute (str): attribute to look for in other attributes.

    #     Returns:
    #         Any: attribute inside specific attributes.

    #     Raises:
    #         AttributeError: if 'attribute' is not found in any of the searched
    #             attributes.

    #     """
    #     if ('slices' in self.__dict__
    #             and attribute.lower() in self.__dict__['slices']):
    #         self.__dict__[attribute.lower()] = value
    #     else:
    #         setattr(self.__dict__['dataset']['full'], attribute, value)

    def __add__(self, other: 'DataSlice') -> None:
        """Adds 'other' to 'dataset'.

        Args:
            other ('DataSlice'): an 'DataSlice' instance.

        """
        self.add(name = other.name, data = other)
        return self

    def __iadd__(self, other: 'DataSlice') -> None:
        """Adds 'other' to 'dataset'.

        Args:
            other ('DataSlice'): an 'DataSlice' instance.

        """
        self.add(name = other.name, data = other)
        return self

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
        if isinstance(self.dataset, pd.DataFrame):
            self.full = DataSlice(data = self.dataset, dataset = self)
            del self.dataset
        return self

    """ Public Methods """

    def add(self, name: str, data: 'DataSlice') -> None:
        """Adds 'data' to 'dataset'.

        Args:
            data ('DataSlice'): an 'DataSlice' instance.

        """
        self.__dict__[name] = DataSlice.create(
            data = data,
            inventory = self.inventory,
            dataset = self)
        return self

    def split_xy(self,
            data: Optional['DataSlice'] = None,
            label: Optional[str] = None) -> None:
        """Splits data into 'x' and 'y' based upon the label passed.

        Args:
            data (Optional['DataSlice']): instance storing a pandas DataFrame.
            label (Optional[str]): name of column to be stored in 'y'.

        """
        if data is None:
            data = self.full
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
        self.active = 'x'
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
            # Tries to apply method to pandas object itself.
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

    # def _infer_type(self, column: pd.Series) -> str:
    #     """Infers column datatype of a single column.

    #     This method is an alternative to default pandas methods which can use
    #     complex datatypes (e.g., int8, int16, int32, int64, etc.) instead of
    #     simple types.

    #     Non-standard python datatypes cannot be inferred.

    #     Args:
    #         column (pd.Series): column for which datatype is sought.

    #     Returns:
    #         str: name of siMpLify proxy datatype name.

    #     """
    #     try:
    #         self.datatypes[column] = self.data.select_dtypes(
    #                 include = [datatype]).columns.to_list()[0]
    #     except AttributeError:
    #         pass
    #     return self

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

    def apply(self, callable: Callable, **kwargs) -> None:
        """Applies 'callable' to 'data' atttribute with **kwargs).

        Args:
            callable (Callable): to be applied to 'data'.
            kwargs: any arguments to be passed to 'callable'.

        """
        self.data = callable(self.data, **kwargs)
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

    def indexify(self,
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