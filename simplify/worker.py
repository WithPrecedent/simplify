"""
.. module:: Worker
:synopsis: generic siMpLify clerk
:publisher: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections.abc
import dataclasses
import importlib
import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import pathos.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

import numpy as np
import pandas as pd

import simplify
from simplify import core
from simplify.core import utilities


@dataclasses.dataclass
class Instructions(core.SimpleLoader):
    """Instructions for 'Worker' construction and usage.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.worker'.
        default_module (Optional[str]): a backup name of module where object to
            use is located (can either be a siMpLify or non-siMpLify module).
            Defaults to 'simplify.worker'.
        worker (Optional[str]): name of Worker object in 'module' to load.
            Defaults to 'Worker'.
        technique (Optional[str]): name of Technique object in 'module' to load.
            Defaults to 'Technique'.
        specialist (Optional[str]): name of Specialist object in 'module' to
            load. Defaults to 'Specialist'.
        finalizer (Optional[str]): name of Finalizer object in 'module' to
            load. Defaults to 'Finalizer'.
        scholar (Optional[str]): name of Scholar object in 'module' to
            load. Defaults to 'Scholar'.
        book (Optional[str]): value to use for 'name' attribute of SimplePlan
            instance that constitutes the 'book'. Defaults to 'book'.
        chapter (Optional[str]): value to use for 'name' attribute of SimplePlan
            instance that constitutes a 'chapter'. Defaults to 'chapter'.
        options (Optional[Union[str, core.SimpleRepository]]): name of a
            SimpleRepository instance with various options available to a
            particular 'Worker' instance or a SimpleRepository instance.
            Defaults to an empty SimpleRepository.
        data (Optional[str]): name of attribute or key in a 'Project' instance
            'books' to use as a data object to apply methods to. Defaults to
            'dataset'.
        import_folder (Optional[str]): name of attribute in 'clerk' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'clerk' which
            contains the path to the default folder for exporting data objects.
            Defaults to 'processed'.
        comparer (Optional[bool]): whether the 'Worker' has a parallel structure
            allowing for comparison of different alternatives (True) or is a
            singular sequence of steps (False). Defaults to False.

    """
    name: Optional[str] = None
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.worker')
    default_module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.worker')
    worker: Optional[str] = dataclasses.field(
        default_factory = lambda: 'Worker')
    technique: Optional[str] = dataclasses.field(
        default_factory = lambda: 'Technique')
    specialist: Optional[str] = dataclasses.field(
        default_factory = lambda: 'Specialist')
    finalizer: Optional[str] = dataclasses.field(
        default_factory = lambda: 'Finalizer')
    scholar: Optional[str] = dataclasses.field(
        default_factory = lambda: 'Scholar')
    book: Optional[str] = dataclasses.field(
        default_factory = lambda: 'book')
    chapter: Optional[str] = dataclasses.field(
        default_factory = lambda: 'chapter')
    options: Optional[Union[str, core.SimpleRepository]] = dataclasses.field(
        default_factory = core.SimpleRepository)
    loadables: Optional[List[str]] = dataclasses.field(
        default_factory = lambda: [
            'options',
            'specialist',
            'finisher',
            'scholar',
            'technique'])
    data: Optional[str] = dataclasses.field(
        default_factory = lambda: 'dataset')
    import_folder: Optional[str] = dataclasses.field(
        default_factory = lambda: 'processed')
    export_folder: Optional[str] = dataclasses.field(
        default_factory = lambda: 'processed')
    comparer: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super().__post_init__()
        self.worker = self.load('worker')(name = self.name)
        for attribute in self.loadables:
            if isinstance(getattr(self, attribute), str):
                setattr(self, attribute, self.load(attribute))
        return self


@dataclasses.dataclass
class Worker(core.SimpleProject):
    """Generic subpackage controller class for siMpLify data projects.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        idea (Optional[Idea]): shared project configuration settings.
        instructions (Optional[Instructions]): an instance with information to
            create and apply the essential components of a Worker. Defaults to
            None.
        auto_draft (Optional[bool]): whether to call the 'draft' method when
            instanced. Defaults to True.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            instanced. Defaults to True.
        auto_apply (Optional[bool]): whether to call the 'apply' method when
            instanced. For auto_apply to have an effect, 'dataset' must also
            be passed. Defaults to False.

    """
    name: Optional[str] = None
    instructions: Optional[Instructions] = None
    idea: Optional[core.Idea] = None
    auto_draft: Optional[bool] = True
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super().__post_init__()
        # Creates instances of helper classes which aid in completing and
        # applying Technique instances.
        for attribute in ['specialist', 'finisher', 'scholar']:
            setattr(self, attribute, getattr(self.instructions, attribute)(
                idea = self.idea,
                instructions = self.instructions))
        # Returns appropriate subclass base on 'comparer' attribute of
        # 'instructions'.
        if self.instructions.comparer:
            return Comparer(
                name = self.name,
                instructions = self.instructions,
                idea = self.idea,
                auto_draft = self.auto_draft,
                auto_publish = self.auto_publish,
                auto_apply = self.auto_apply)
        else:
            return Sequencer(
                name = self.name,
                instructions = self.instructions,
                idea = self.idea,
                auto_draft = self.auto_draft,
                auto_publish = self.auto_publish,
                auto_apply = self.auto_apply)

    """ Public Methods """

    def publish(self, book: core.SimplePlan) -> core.SimplePlan:
        """Finalizes each technique in each chapter of 'book'.

        Args:
            book ('SimplePlan'): iterable storing different chapter instances.

        Returns:
            SimplePlan: with techniques finalized.

        """
        new_chapters = []
        for chapter in book:
            chapter.contents = [
                self.specialist.apply(technique = t) for t in chapter]
        book.contents = new_chapters
        return book

    def apply(self,
            library: core.SimpleRepository,
            data: core.Dataset,
            **kwargs) -> (core.SimpleRepository, core.Dataset):
        """

        Args:


        Returns:


        """
        # Gets appropriate data based upon 'data' attribute of 'instructions'.
        data_to_use = self._set_data(library = library, data = data)
        # Finalizes each 'Technique' instance and instances each 'algorithm'
        # with corresponding 'parameters'.
        for chapter in self.library[self.name]:
            new_steps = []
            for technique in chapter:
                new_steps.append(self.finisher.apply(
                    book = self.library[self.name],
                    data = data_to_use))
            chapter.contents = new_steps
        # Applies each 'Technique' instance to 'data_to_use'.
        self.library[self.name] = self.specialist.apply(
            book = self.library[self.name],
            data = data_to_use,
            **kwargs)
        return library, data

    """ Overview Property """

    @property
    def overview(self) -> Dict[str, List[str]]:
        """Returns snapshot of current state of selected options.

        Returns:
            Dict[str, List[str]]: keys are steps and values are lists of
                selected options.

        """
        try:
            return self._overview
        except AttributeError:
            self._overview = self._get_overview()
            return self._overview

    @overview.setter
    def overview(self, overview: Dict[str, List[str]]) -> None:
        """Sets snapshot of selected options.

        Setting 'overview' will affect other methods which use 'overview' to
        identify which options have been selected.

        Args:
            overview (Dict[str, List[str]]): keys are steps and values are lists
                of selected options.

        """
        if (isinstance(overview, dict)
                and all(isinstance(v, list) for v in overview.values())):
            self._overview = overview
        else:
            raise TypeError(f'overview must be dict of lists')
        return self

    @overview.deleter
    def overview(self) -> None:
        """Sets snapshot of selected options to an empty dictionary.

        There are few, if any reasons, to use the 'overview' deleter. It is
        included in case a user wants the option to clear out current selections
        and add more manually.

        """
        self._overview = {}
        return self

    """ Private Methods """

    def _get_overview(self) -> Dict[str, List[str]]:
        """Creates dictionary with techniques for each step.

        Returns:
            Dict[str, Dict[str, List[str]]]: dictionary with keys of steps and
                values of lists of techniques.

        """
        steps = self.idea.get_steps(section = self.instructions.name)
        outline = {}
        for step in steps:
            outline[step] = self.idea.get_techniques(
                section = self.instructions.name,
                step = step)
        return outline

    def _set_data(self,
            library: core.SimpleRepository,
            data: core.Dataset) -> Union[core.SimpleComponent, core.Dataset]:
        """Returns appropriate data object based upon 'instructions'.

        Args:

        Returns:

        """
        if self.instructions.data in ['dataset']:
            return data
        else:
            return library[self.instructions.data]


@dataclasses.dataclass
class Comparer(Worker):
    """Generic subpackage controller class for siMpLify data projects.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        idea (Optional[Idea]): shared project configuration settings.
        instructions (Optional[Instructions]): an instance with information to
            create and apply the essential components of a Worker. Defaults to
            None.
        auto_draft (Optional[bool]): whether to call the 'draft' method when
            instanced. Defaults to True.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            instanced. Defaults to True.
        auto_apply (Optional[bool]): whether to call the 'apply' method when
            instanced. For auto_apply to have an effect, 'dataset' must also
            be passed. Defaults to False.

    """
    name: Optional[str] = None
    instructions: Optional[Instructions] = None
    idea: Optional[core.Idea] = None
    auto_draft: Optional[bool] = True
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super(core.SimpleProject).__post_init__()
        return self

    """ Public Methods """

    def draft(self) -> core.SimplePlan:
        """Drafts a book with a parallel chapter structure.

        Returns:
            'SimplePlan': configured to spefications in 'instructions'.

        """
        # Creates a 'SimplePlan' instance to store other 'SimplePlan' instances.
        book = core.SimplePlan(name = self.instructions.book)
        # Creates list of steps from 'outline'.
        steps = list(self.overview.keys())
        # Creates 'possible' list of lists of 'techniques'.
        possible = list(self.overview.values())
        # Creates a list of lists of the Cartesian product of 'possible'.
        combinations = list(map(list, itertools.product(*possible)))
        # Creates a 'chapter' for each combination of techniques and adds that
        # 'chapter' to 'book'.
        for i, techniques in enumerate(combinations):
            chapter = core.SimplePlan(name = f'{self.instructions.chapter}_{i}')
            step_techniques = tuple(zip(steps, techniques))
            for technique in step_techniques:
                technique = self.instructions.technique.load()(
                    name = technique[0],
                    technique = technique[1])
                technique = self.specialist.draft(technique = technique)
                chapter.add(contents = technique)
            book.add(contents = chapter)
        return book


@dataclasses.dataclass
class Sequencer(Worker):
    """Generic subpackage controller class for siMpLify data projects.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        idea (Optional[Idea]): shared project configuration settings.
        instructions (Optional[Instructions]): an instance with information to
            create and apply the essential components of a Worker. Defaults to None.
        auto_draft (Optional[bool]): whether to call the 'draft' method when
            instanced. Defaults to True.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            instanced. Defaults to True.
        auto_apply (Optional[bool]): whether to call the 'apply' method when
            instanced. For auto_apply to have an effect, 'dataset' must also
            be passed. Defaults to False.

    """
    name: Optional[str] = None
    instructions: Optional[Instructions] = None
    idea: Optional[core.Idea] = None
    auto_draft: Optional[bool] = True
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super(core.SimpleProject).__post_init__()
        return self

    """ Public Methods """

    def draft(self) -> core.SimplePlan:
        """Drafts a book with a serial chapter structure.

        Returns:
            'SimplePlan': configured to spefications in 'instructions'.

        """
        # Creates a 'SimplePlan' instance to store other 'SimplePlan' instances.
        book = core.SimplePlan(name = self.instructions.book)
        # Creates a 'chapter' for each step in 'overview'.
        for step, techniques in self.overview.items():
            chapter = core.SimplePlan(name = step)
            for technique in techniques:
                technique = self.instructions.technique(
                    name = step,
                    technique = technique)
                chapter.add(contents = technique)
            book.add(contents = chapter)
        return book


@dataclasses.dataclass
class Specialist(core.SimpleHandler):
    """Constructs 'Technique' with an 'algorithm' and 'parameters'.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        idea (Optional[Idea]): shared project configuration settings.
        instructions (Optional[Instructions]): an instance with information to
            create and apply the essential components of a Worker. Defaults to
            None.

    """
    name: Optional[str] = None
    idea: Optional[core.Idea] = None
    instructions: Optional[Instructions] = None

    """ Public Methods """

    def apply(self, technique: core.Technique) -> core.Technique:
        """Finalizes 'Technique' instance from 'step'.

        Args:
            step (Tuple[str, str]): information needed to create a 'Technique'
                instance.

        Returns:
            'Technique': instance ready for application.

        """
        if technique.technique in ['none']:
            return technique
        else:
            technique.load('algorithm')
            return self._get_parameters(technique = technique)

    """ Private Methods """

    def _get_technique(self,
            technique: core.Technique,
            step: Tuple[str, str]) -> core.Technique:
        """Finalizes 'technique'.

        Args:
            technique (Technique): an instance for parameters to be added to.

        Returns:
            Technique: instance with parameters added.

        """
        technique.step = step[0]
        if technique.module and technique.algorithm:
            technique.algorithm = technique.load('algorithm')
        return self._get_parameters(technique = technique)

    def _get_parameters(self, technique: 'Technique') -> 'Technique':
        """Finalizes 'parameters' for 'technique'.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        parameter_types = ['idea', 'selected', 'required', 'runtime']
        # Iterates through types of 'parameter_types'.
        for parameter_type in parameter_types:
            try:
                technique = getattr(self, '_'.join(
                    ['_publish', parameter_type]))(technique = technique)
            except (TypeError, AttributeError):
                pass
        return technique

    def _get_idea(self, technique: 'Technique') -> 'Technique':
        """Acquires parameters from 'Idea' instance.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        try:
            technique.parameters.update(
                self.idea['_'.join([technique.name, 'parameters'])])
        except KeyError:
            try:
                technique.parameters.update(
                    self.idea['_'.join([technique.step, 'parameters'])])
            except AttributeError:
                pass
        return technique

    def _get_selected(self, technique: 'Technique') -> 'Technique':
        """Limits parameters to those appropriate to the 'technique'.

        If 'technique.selected' is True, the keys from 'technique.defaults' are
        used to select the final returned parameters.

        If 'technique.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        if technique.selected:
            if isinstance(technique.selected, list):
                parameters_to_use = technique.selected
            else:
                parameters_to_use = list(technique.default.keys())
            new_parameters = {}
            for key, value in technique.parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            technique.parameters = new_parameters
        return technique

    def _get_required(self, technique: 'Technique') -> 'Technique':
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        try:
            technique.parameters.update(technique.required)
        except TypeError:
            pass
        return technique

    def _get_search(self, technique: 'Technique') -> 'Technique':
        """Separates variables with multiple options to search parameters.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        technique.parameter_space = {}
        new_parameters = {}
        for parameter, values in technique.parameters.items():
            if isinstance(values, list):
                if any(isinstance(i, float) for i in values):
                    technique.parameter_space.update(
                        {parameter: uniform(values[0], values[1])})
                elif any(isinstance(i, int) for i in values):
                    technique.parameter_space.update(
                        {parameter: randint(values[0], values[1])})
            else:
                new_parameters.update({parameter: values})
        technique.parameters = new_parameters
        return technique

    def _get_runtime(self, technique: 'Technique') -> 'Technique':
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        try:
            for key, value in technique.runtime.items():
                try:
                    technique.parameters.update(
                        {key: getattr(self.idea['general'], value)})
                except AttributeError:
                    raise AttributeError(' '.join(
                        ['no matching runtime parameter', key, 'found']))
        except (AttributeError, TypeError):
            pass
        return technique


@dataclasses.dataclass
class Finisher(core.SimpleHandler):
    """Finalizes 'Technique' instances with data-dependent parameters.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        idea (Optional[Idea]): shared project configuration settings.
        instructions (Optional[Instructions]): an instance with information to
            create and apply the essential components of a Worker. Defaults to
            None.

    """
    name: Optional[str] = None
    idea: Optional[core.Idea] = None
    instructions: Optional[Instructions] = None

    """ Public Methods """

    def apply(self,
            book: core.SimplePlan,
            data: Union[core.Dataset, core.SimplePlan]) -> core.SimplePlan:
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
        new_chapters = [
            self._finalize_techniques(chapter = chapter, data = data)
            for chapter in book.chapters]

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
            if technique.name not in ['none']:
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


@dataclasses.dataclass
class Scholar(core.SimpleHandler):
    """Base class for applying 'Technique' instances to data.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        idea (Optional[Idea]): shared project configuration settings.
        instructions (Optional[Instructions]): an instance with information to
            create and apply the essential components of a Worker. Defaults to
            None.

    """
    name: Optional[str] = None
    instructions: Optional[Instructions] = None
    idea: Optional[core.Idea] = None

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