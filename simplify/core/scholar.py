"""
.. module:: scholar
:synopsis: generic siMpLify application classes
:publisher: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections.abc
import dataclasses
import importlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import pathos.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

import numpy as np
import pandas as pd

from simplify.core import base
from simplify.core import utilities


@dataclasses.dataclass
class Scholar(base.SimpleCreator):
    """Base class for applying 'Book' instances to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self = self.idea.apply(instance = self)
        # Creates 'Finisher' instance to finalize 'Technique' instances.
        self.finisher = Finisher(worker = self.worker)
        # Creates 'Specialist' instance to apply 'Technique' instances.
        self.specialist = Specialist(worker = self.worker)
        # Creates 'Parallelizer' instance to apply 'Chapter' instances, if the
        # option to parallelize has been selected.
        if self.parallelize:
            self.parallelizer = Parallelizer(idea = self.idea)
        return self

    """ Private Methods """

    def _set_data(self,
            project: 'Project',
            data: 'Dataset') -> Union['Dataset', 'Book']:
        """Returns 'data' appropriate to 'worker'.

        Args:
            project ('Project'): instance with stored 'Book' instances.
            data ('Dataset'): primary instance used by 'project'.

        Returns:
            Union['Dataset', 'Book]: primary dataset or 'Book' instance from
                'project' depending upon the 'data' attribute' of 'worker'.

        """
        if self.worker.data in ['dataset']:
            return data
        else:
            return project[self.worker.data]

    """ Core siMpLify Methods """

    def apply(self,
            project: 'Project',
            data: 'Dataset',
            **kwargs) -> ('Project', 'Dataset'):
        """Applies 'Book' instance in 'project' to 'data' or other stored books.

        Args:
            worker (str): key to 'Book' instance to apply in 'project'.
            project ('Project): instance with stored 'Book' instances to apply
                or to have other 'Book' instances applied to.
            data (Optional[Union['Dataset', 'Book']]): a data source 'Book'
                instances in 'project' to potentially be applied.
            kwargs: any additional parameters to pass.

        Returns:
            Tuple('Project', 'Data'): instances with any necessary modifications
                made.

        """
        # Gets appropriate data based upon 'data' attribute of 'worker'.
        data_to_use = self._set_data(project = project, data = data)
        # Finalizes each 'Technique' instance and instances each 'algorithm'
        # with corresponding 'parameters'.
        project[self.worker.name] = self.finisher.apply(
            book = project[self.worker.name],
            data = data_to_use)
        # Applies each 'Technique' instance to 'data_to_use'.
        project[self.worker.name] = self.specialist.apply(
            book = project[self.worker.name],
            data = data_to_use)
        return project


@dataclasses.dataclass
class Finisher(SimpleCreator):
    """Finalizes 'Technique' instances with data-dependent parameters.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self = self.idea.apply(instance = self)
        return self

    """ Private Methods """

    def _finalize_chapters(self, book: 'Book', data: 'Dataset') -> 'Book':
        """Finalizes 'Chapter' instances in 'Book'.

        Args:
            book ('Book'): instance containing 'chapters' with 'techniques' that
                have 'data_dependent' and/or 'conditional' 'parameters' to
                add.
            data ('Dataset): instance with potential information to use to
                finalize 'parameters' for 'book'.

        Returns:
            'Book': with any necessary modofications made.

        """
        new_chapters = []
        for chapter in book.chapters:
            new_chapters.append(
                self._finalize_techniques(manuscript = chapter, data = data))
        book.chapters = new_chapters
        return book

    def _finalize_techniques(self,
            manuscript: Union['Book', 'Chapter'],
            data: ['Dataset', 'Book']) -> Union['Book', 'Chapter']:
        """Subclasses may provide their own methods to finalize 'techniques'.

        Args:
            manuscript (Union['Book', 'Chapter']): manuscript containing
                'techniques' to apply.
            data (['Dataset', 'Book']): instance with information used to
                finalize 'parameters' and/or 'algorithm'.

        Returns:
            Union['Book', 'Chapter']: with any necessary modofications made.

        """
        new_techniques = []
        for technique in manuscript.techniques:
            if not technique.name in ['none']:
                new_technique = self._add_conditionals(
                    manuscript = manuscript,
                    technique = technique,
                    data = data)
                new_technique = self._add_data_dependent(
                    technique = technique,
                    data = data)
                new_techniques.append(self._add_parameters_to_algorithm(
                    technique = technique))
        manuscript.techniques = new_techniques
        return manuscript

    def _add_conditionals(self,
            manuscript: 'Book',
            technique: 'Technique',
            data: Union['Dataset', 'Book']) -> 'Technique':
        """Adds any conditional parameters to a 'Technique' instance.

        Args:
            manuscript ('Book'): Book instance with algorithms to apply to 'data'.
            technique ('Technique'): instance with parameters which can take
                new conditional parameters.
            data (Union['Dataset', 'Book']): a data source which might
                contain information for condtional parameters.

        Returns:
            'technique': instance with any conditional parameters added.

        """
        try:
            if technique is not None:
                return getattr(manuscript, '_'.join(
                    ['_add', technique.name, 'conditionals']))(
                        technique = technique,
                        data = data)
        except AttributeError:
            return technique

    def _add_data_dependent(self,
            technique: 'Technique',
            data: Union['Dataset', 'Book']) -> 'Technique':
        """Completes parameter dictionary by adding data dependent parameters.

        Args:
            technique ('Technique'): instance with information about data
                dependent parameters to add.
            data (Union['Dataset', 'Book']): a data source which contains
                'data_dependent' variables.

        Returns:
            'Technique': with any data dependent parameters added.

        """
        if technique is not None and technique.data_dependent is not None:

            for key, value in technique.data_dependent.items():
                try:
                    technique.parameters.update({key: getattr(data, value)})
                except KeyError:
                    print('no matching parameter found for', key, 'in data')
        return technique

    def _add_parameters_to_algorithm(self,
            technique: 'Technique') -> 'Technique':
        """Instances 'algorithm' with 'parameters' in 'technique'.

        Args:
            technique ('Technique'): with completed 'algorith' and 'parameters'.

        Returns:
            'Technique': with 'algorithm' instanced with 'parameters'.

        """
        if technique is not None:
            try:
                technique.algorithm = technique.algorithm(
                    **technique.parameters)
            except AttributeError:
                try:
                    technique.algorithm = technique.algorithm(
                        technique.parameters)
                except AttributeError:
                    technique.algorithm = technique.algorithm()
            except TypeError:
                try:
                    technique.algorithm = technique.algorithm()
                except TypeError:
                    pass
        return technique

    """ Core siMpLify Methods """

    def apply(self, book: 'Book', data: Union['Dataset', 'Book']) -> 'Book':
        """Applies 'Book' instance in 'project' to 'data' or other stored books.

        Args:
            book ('Book'): instance with stored 'Technique' instances (either
                stored in the 'techniques' or 'chapters' attributes).
            data ([Union['Dataset', 'Book']): a data source with information to
                finalize 'parameters' for each 'Technique' instance in 'book'

        Returns:
            'Book': with 'parameters' for each 'Technique' instance finalized
                and connected to 'algorithm'.

        """
        if hasattr(book, 'techniques'):
            book = self._finalize_techniques(manuscript = book, data = data)
        else:
            book = self._finalize_chapters(book = book, data = data)
        return book


@dataclasses.dataclass
class Specialist(SimpleCreator):
    """Base class for applying 'Technique' instances to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self = self.idea.apply(instance = self)
        return self

    """ Private Methods """

    def _apply_chapters(self,
            book: 'Book',
            data: Union['Dataset', 'Book']) -> 'Book':
        """Applies 'chapters' in 'Book' instance in 'project' to 'data'.

        Args:
            book ('Book'): instance with stored 'Chapter' instances.
            data ('Dataset'): primary instance used by 'project'.

        Returns:
            'Book': with modifications made and/or 'data' incorporated.

        """
        new_chapters = []
        for i, chapter in enumerate(book.chapters):
            if self.verbose:
                print('Applying', chapter.name, str(i + 1), 'to', data.name)
            new_chapters.append(self._apply_techniques(
                manuscript = chapter,
                data = data))
        book.chapters = new_chapters
        return book

    def _apply_techniques(self,
            manuscript: Union['Book', 'Chapter'],
            data: Union['Dataset', 'Book']) -> Union['Book', 'Chapter']:
        """Applies 'techniques' in 'manuscript' to 'data'.

        Args:
            manuscript (Union['Book', 'Chapter']): instance with stored
                'techniques'.
            data ('Dataset'): primary instance used by 'manuscript'.

        Returns:
            Union['Book', 'Chapter']: with modifications made and/or 'data'
                incorporated.

        """
        for technique in manuscript.techniques:
            if self.verbose:
                print('Applying', technique.name, 'to', data.name)
            if isinstance(data, Dataset):
                data = technique.apply(data = data)
            else:
                for chapter in data.chapters:
                    manuscript.chapters.append(technique.apply(data = chapter))
        if isinstance(data, Dataset):
            setattr(manuscript, 'data', data)
        return manuscript

    """ Core siMpLify Methods """

    def apply(self, book: 'Book', data: Union['Dataset', 'Book']) -> 'Book':
        """Applies 'Book' instance in 'project' to 'data' or other stored books.

        Args:
            book ('Book'): instance with stored 'Technique' instances (either
                stored in the 'techniques' or 'chapters' attributes).
            data ([Union['Dataset', 'Book']): a data source with information to
                finalize 'parameters' for each 'Technique' instance in 'book'

        Returns:
            'Book': with 'parameters' for each 'Technique' instance finalized
                and connected to 'algorithm'.

        """
        if hasattr(book, 'techniques'):
            book = self._apply_techniques(manuscript = book, data = data)
        else:
            book = self._apply_chapters(book = book, data = data)
        return book


@dataclasses.dataclass
class Parallelizer(SimpleCreator):
    """Applies techniques using one or more CPU or GPU cores.

    Args:
        idea ('Idea'): shared 'Idea' instance with project settings.

    """
    idea: 'Idea'

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        return self

    """ Private Methods """

    def _apply_gpu(self,
            book: 'Book',
            data: Union['Dataset', 'Book'],
            method: Callable) -> 'Book':
        """Applies objects in 'book' to 'data'

        Args:
            book ('Book'): siMpLify class instance to be
                modified.
            data (Optional[Union['Dataset', 'Book']]): an
                Dataset instance containing external data or a published
                Book. Defaults to None.
            kwargs: any additional parameters to pass to a related
                Book's 'apply' method.

        Raises:
            NotImplementedError: until dynamic GPU support is added.

        """
        raise NotImplementedError(
            'GPU support outside of modeling is not yet supported')

    def _apply_multi_core(self,
            book: 'Book',
            data: Union['Dataset', 'Book'],
            method: Callable) -> 'Book':
        """Applies 'method' to 'data' using multiple CPU cores.

        Args:
            book ('Book'): siMpLify class instance with Chapter instances to
                parallelize.
            data (Union['Dataset', 'Book']): an instance containing data to
                be modified.
            method (Callable): method to parallelize.

        Returns:
            'Book': with its iterable applied to data.

        """
        with mp.Pool() as pool:
            pool.starmap(method, arguments)
        pool.close()
        return self

    """ Core siMpLify Methods """

    def apply_chapters(self,
            book: 'Book',
            data: Union['Dataset', 'Book'],
            method: Callable) -> 'Book':
        """Applies 'method' to 'data'.

        Args:
            book ('Book'): siMpLify class instance with Chapter instances to
                parallelize.
            data (Union['Dataset', 'Book']): an instance containing data to
                be modified.
            method (Callable): method to parallelize.

        Returns:
            'Book': with its iterable applied to data.

        """
        arguments = []
        for key, chapter in book.chapters.items():
            arguments.append((chapter, data))
        results = []
        chapters_keys = list(book.chapters.keys())
        with mp.Pool() as pool:
            results.append[pool.map(method, arguments)]
        pool.close()
        pool.join()
        pool.clear()
        book.chapters = dict(zip(chapters_keys, results))
        return book

    def apply_data(self,
            data: 'Data',
            method: Callable) -> 'Data':
        """Applies 'method' to 'data' across several cores.

        Args:
            data ('Data'): instance with a stored pandas DataFrame.
            method (Callable): process method or function to apply to 'data'.

        Returns:
            'Data': with 'method' applied.

        """
        dfs = np.array_split(data.data, mp.cpu_count(), axis = 0)
        pool = mp.Pool()
        data.data = np.vstack(pool.map(method, dfs))
        pool.close()
        pool.join()
        pool.clear()
        return data
