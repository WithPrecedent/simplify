"""
wrangler.components:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from types import ModuleType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import simplify
from . import base



@dataclasses.dataclass
class Sow(SimpleIterable):
    """Acquires and performs basic preparation of data sources.

    Args:
        steps(dict): dictionary containing keys of WranglerTechnique names (strings)
            and values of WranglerTechnique class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    steps: object = None
    name: str = 'sower'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        self.needed_parameters = {'convert': ['file_in', 'file_out',
                                                 'method'],
                                  'download': ['file_url', 'file_name'],
                                  'scrape': ['file_url', 'file_name'],
                                  'split': ['in_folder', 'out_folder',
                                                'method']}
        if self.step in ['split']:
            self.import_folder = 'raw'
            self.export_folder = 'interim'
        else:
            self.import_folder = 'external'
            self.export_folder = 'external'
        return self

    def publish(self, dataset):
        self.algorithm.implement(dataset)
        return dataset


@dataclasses.dataclass
class Scrape(WranglerTechnique):
    """Scrapes data from a website.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'converter'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self


    def publish(self, dataset):
        file_path = os.path.join(self.clerk.external, self.file_name)
        return self


@dataclasses.dataclass
class Download(WranglerTechnique):
    """Acquires data from an online source.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'downloader'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def publish(self):
        return self

    def publish(self, dataset):
        """Downloads file from a URL if the file is available."""
        file_path = os.path.join(self.clerk.external,
                                 self.file_name)
        file_response = requests.get(self.file_url)
        with open(file_path, 'wb') as file:
            file.write(file_response.content)
        return self


@dataclasses.dataclass
class Harvest(SimpleIterable):
    """Extracts data from text or other sources.

    Args:
        steps(dict): dictionary containing keys of WranglerTechnique names (strings)
            and values of WranglerTechnique class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    steps: object = None
    name: str = 'harvester'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def _publish_organize(self, key):
        file_path = os.path.join(self.clerk.techniques,
                                 'organizer_' + key + '.csv')
        self.parameters = {'step': self.step,
                           'file_path': file_path}
        algorithm = self.workers[self.step](**self.parameters)
        self._set_columns(algorithm)
        return algorithm

    def _publish_parse(self, key):
        file_path = os.path.join(self.clerk.techniques,
                                 'parser_' + key + '.csv')
        self.parameters = {'step': self.step,
                           'file_path': file_path}
        algorithm = self.workers[self.step](**self.parameters)
        return algorithm

    def draft(self) -> None:
        return self

    def _set_columns(self, algorithm):
        prefix = algorithm.matcher.section_prefix
        if not hasattr(self, 'columns'):
            self.columns = []
        new_columns = list(algorithm.expressions.values())
        new_columns = [prefix + '_' + column for column in self.columns]
        self.columns.extend(new_columns)
        return self

    def _implement_organize(self, dataset, algorithm):
        dataset.df, dataset.source = algorithm.implement(
                df = dataset.df, source = dataset.source)
        return dataset

    def _implement_parse(self, dataset, algorithm):
        dataset.df = algorithm.implement(df = dataset.df,
                                         source = dataset.source)
        return dataset

    def publish(self):
        for key in self.parameters:
            if hasattr(self, '_publish_' + self.step):
                algorithm = getattr(
                        self, '_publish_' + self.step)(key = key)
            else:
                algorithm = getattr(self, '_publish_generic_list')(key = key)
            self.algorithms.append(algorithm)
        return self
    
    
@dataclasses.dataclass
class Bale(SimpleIterable):
    """Class for combining different datasets."""
    step: object = None
    parameters: object = None
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        self.needed_parameters = {'merger': ['index_columns', 'merge_type']}
        return self

    def publish(self):
        self.algorithm = self.workers[self.step](**self.parameters)
        return self

    def publish(self, dataset):
        data = self.algorithm.implement(dataset)
        return dataset


@dataclasses.dataclass
class Clean(SimpleIterable):
    """Cleans, munges, and parsers data using fast, vectorized methods.

    Args:
        steps(dict): dictionary containing keys of WranglerTechnique names (strings)
            and values of WranglerTechnique class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    steps: object = None
    name: str = 'cleaner'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        return self

    def _implement_combiner(self, dataset):
        data = self.algorithm.implement(dataset)
        return dataset

    def _implement_keyword(self, dataset):
        dataset.df = self.algorithm.implement(dataset.df)
        return dataset

    def publish(self, dataset):
        data = getattr(self, '_implement_' + self.step)(dataset)
        return dataset


@dataclasses.dataclass
class Combine(WranglerTechnique):
    """Combines features into new features.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'combiner'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init()
        return self

    def _combine_all(self, dataset):
        dataset.df[self.parameters['out_column']] = np.where(
                np.all(dataset.df[self.parameters['in_columns']]),
                True, False)
        return dataset

    def _combine_any(self, dataset):
        dataset.df[self.parameters['out_column']] = np.where(
                np.any(dataset.df[self.parameters['in_columns']]),
                True, False)
        return dataset

    def _dict(self, dataset):
        dataset.df[self.parameters['out_column']] = (
                dataset.df[self.parameters['in_columns']].map(
                        self.method))
        return dataset

    def draft(self) -> None:
        self._options = SimpleRepository(contents = {'all': self._combine_all,
                        'any': self._combine_any,
                        'dict': self._dict}
        if isinstance(self.method, str):
            self.algorithm = self.workers[self.method]
        else:
            self.algorithm = self._dict
        return self

    def publish(self, dataset):
        self.data = self.algorithm(dataset)
        return dataset
        

@dataclasses.dataclass
class Deliver(SimpleIterable):
    """Makes final structural changes to data before analysis.

    Args:
        steps(dict): dictionary containing keys of WranglerTechnique names (strings)
            and values of WranglerTechnique class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    steps: object = None
    name: str = 'delivery'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def _publish_shapers(self, harvest):
        self.algorithm = self.workers[self.step](**self.parameters)
        return self

    def _publish_streamliners(self, harvest):
        self.algorithm = self.workers[self.step](**self.parameters)
        return self

    def draft(self) -> None:
        self.needed_parameters = {'shapers': ['shape_type', 'stubs',
                                               'id_column', 'values',
                                               'separator'],
                                  'streamliners': ['method']}
        return self

    def publish(self, dataset):
        data = self.algorithm.implement(dataset)
        return dataset


@dataclasses.dataclass
class Divide(WranglerTechnique):
    """Divides data source files so that they can be loaded in memory.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'converter'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def publish(self, dataset):
        return self


@dataclasses.dataclass
class Merge(WranglerTechnique):
    """Merges data sources together.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'encoder'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        return self

    def draft(self) -> None:
        self._options = SimpleRepository(contents = {}
        return self

    def publish(self, dataset, sources):
        return dataset
        

@dataclasses.dataclass
class Reshape(WranglerTechnique):
    """Reshapes a DataFrame to wide or long form.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance and
            other methods and classes in the siMpLify package..
        auto_draft (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'scaler'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        return self

    def _long(self, df):
        """A simple wrapper method for pandas wide_to_long method using more
        intuitive parameter names than 'i' and 'j'.
        """
        df = (pd.wide_to_long(df,
                              stubnames = self.stubs,
                              i = self.id_column,
                              j = self.values,
                              sep = self.separator).reset_index())
        return df

    def _wide(self, df):
        """A simple wrapper method for pandas pivot method named as
        corresponding method to reshape_long.
        """
        df = (df.pivot(index = self.id_column,
                       columns = self.stubs,
                       values = self.values).reset_index())
        return df


    def publish(self, dataset):
        dataset.df = getattr(self, '_' + self.shape_type)(dataset.df)
        return dataset
        

@dataclasses.dataclass
class Streamline(WranglerTechnique):
    """Combines, divides, and otherwise prepares features for analysis.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance and
            other methods and classes in the siMpLify package..
        auto_draft (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'scaler'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        return self

    def publish(self, dataset):
        data = self.method(dataset)
        return dataset
        

@dataclasses.dataclass
class Supplement(WranglerTechnique):
    """Adds new data to similarly structured DataFrame.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'supplementer'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        return self

    def draft(self) -> None:
        self._options = SimpleRepository(contents = {}
        return self

    def publish(self, dataset, sources):
        return dataset