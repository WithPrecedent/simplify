"""
.. module:: editors
:synopsis: constructs and applies books, chapters, and techniques
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.base import SimpleEditor
from simplify.core.base import SimpleProgression
from simplify.core.utilities import listify
from simplify.core.validators import DataValidator


@dataclass
class Author(SimpleEditor):
    """ Drafts a skeleton Book instance.

    Args:
        project ('Project'): a related Project instance.
        worker (str): name of the key to the Worker instance in the 'workers'
            attribute of 'project' for which this Class should draft a Book
            instance.

    """
    project: 'Project'
    worker: str

    """ Private Methods """

    def _draft_book(self) -> 'Book':
        """Creates initial, empty Book instance.

        Returns:
            Book: instance with only 'name' set.

        """
        try:
            # Checks to see if a matching Book instance was already created.
            return self.project.library[self.worker]
        except KeyError:
            # Loads a Book class based upon 'worker' attributes.
            book = self.project.workers[self.worker].load('book')
            # Creates an empty class instance of a Book.
            return book(name = self.worker)

    def _draft_steps(self) -> List[str]:
        """Creates a list of 'steps' from 'idea'."""
        # Checks existing 'worker' to see if 'steps' exists.
        if self.workers[self.worker].steps:
            return self.workers[self.worker].steps
        else:
            try:
                # Attempts to get 'steps' from 'idea'.
                return listify(self.project.idea[self.worker]['_'.join(
                    [self.worker, 'steps'])])
            except KeyError:
                return []
            return self

    def _draft_options(self) -> 'SimpleProgression':
        """Gets options for creating a Book contents.

        Returns:
            'SimpleProgression': with coores

        """
        module = self.project.workers[self.worker].module
        try:
            options = getattr(import_module(module), 'get_options')(
                idea = self.project.idea)
        except AttributeError:
            options = getattr(import_module(module), 'DEFAULT_OPTIONS')
        return SimpleProgression(options = options)

    def _draft_techniques(self) -> Dict[str, List[str]]:
        """Creates 'techniques' for a Book.

        Returns:
            Dict[str, List[str]]: possible techniques for each step in a Book
                instance.

        """
        try:
            # Checks to see if 'techniques' already exists.
            return self.project.workers[self.worker].techniques
        except (KeyError, AttributeError):
            techniques = {}
            for step in self.project.workers[self.worker].steps:
                try:
                    techniques[step] = listify(self.project.idea
                        [self.worker]['_'.join([step, 'techniques'])])
                except KeyError:
                    techniques[step] = ['none']
            return techniques

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Drafts initial attributes and settings of a Book instance. """
        # Adds 'Book' to 'library'.
        self.project.library[self.worker] = self._draft_book()
        # Adds 'steps' to 'workers'.
        self.project.workers[self.worker].steps = self._draft_steps()
        # Adds 'techniques' to 'workers'.
        self.project.workers[self.worker].techniques = self._draft_techniques()
        # Adds 'steps' to 'workers'.
        self.project.workers[self.worker].options = self._draft_options()
        return self


@dataclass
class Publisher(SimpleEditor):
    """Finalizes Book instances.

    Args:
        project ('Project'): a related Project instance.
        worker (str): name of the key to the Worker instance in the 'workers'
            attribute of 'project' for which this Class should draft a Book
            instance.

    """
    project: 'Project'
    worker: str

    def __post_init__(self) -> None:
        """Creates 'Contributor' and 'Researcher' instances."""
        self.contributor = Contributor(
            project = self.project,
            worker = self.worker)
        self.researcher = Researcher(
            project = self.project,
            worker = self.worker)
        return self

    """ Private Methods """

    def _publish_techniques(self, book: 'Book') -> 'Book':
        """Publishes instanced 'techniques' for a Book instance.

        Returns:
            Book instance.

        """
        techniques = self.project.workers[self.worker].techniques
        for step, techniques in techniques.items():
            for technique in techniques:
                book.techniques[step] = self.researcher.publish(
                    step = step,
                    technique = technique)
        return book

    def _publish_chapters(self, book: 'Book') -> 'Book':
        """
        """
        # Creates a list of lists of possible techniques.
        possible = list(self.project.workers[self.worker].techniques.values())
        # Converts 'possible' to a list of the Cartesian product.
        plans = list(map(list, product(*possible)))
        # Creates Chapter instance for every combination of step techniques.
        for i, techniques in enumerate(plans):
            book.chapters.add(
                self.contributor.publish(
                    book = book,
                    number = i,
                    techniques = techniques))
        return book

    """ Core siMpLify Methods """

    def publish(self, worker: str) -> None:
        """Finalizes Book instance, making all changes before application.

        Args:
            worker (str): name of Book in the 'library' or 'project'.

        """
        book = self.project.library[self.worker]
        # Creates an 'options' attribute to allow publishing of 'techniques'.
        book = self._publish_techniques(book = book)
        # Creates 'chapters' for 'book'.
        self.project.library[self.worker] = self._publish_chapters(book = book)
        return self


@dataclass
class Researcher(SimpleEditor):
    """Creates Technique instances for a Book instance.

    Args:
        project ('Project'): a related Project instance.
        worker (str): name of the key to the Worker instance in the 'workers'
            attribute of 'project' for which this Class should draft a Book
            instance.

    """
    project: 'Project'
    worker: str

    def __post_init__(self) -> None:
        # Declares possible 'parameter_types'.
        self.parameter_types = [
            'idea',
            'selected',
            'required',
            # 'search',
            'runtime',
            'conditional']
        # Declares 'options' for specific 'worker'.
        self.options = self.project.workers[self.worker].options
        super().__post_init__()
        return self

    """ Private Methods """

    def _publish_parameters(self,
            outline: 'Algorithm') -> Dict[str, Any]:
        """Creates 'parameters' for a 'Technique'.


        """
        # Iterates through types of 'parameter_types'.
        for parameter in self.parameter_types:
            if parameter in ['conditional']:
                getattr(chapter, chapter.iterable)[step] = (
                    self._publish_conditional(
                        book = book,
                        technique = technique))
            else:
                getattr(chapter, chapter.iterable)[step] = (
                    getattr(self, '_'.join(
                        ['_publish', parameter]))(technique = technique))

    def _publish_idea(self, technique: 'Technique') -> 'Technique':
        """Acquires parameters from Idea instance, if no parameters exist.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            'Technique': instance with modifications made.

        """
        if not technique.parameters:
            try:
                technique.parameters.update(self.project.idea.configuration['_'.join(
                    [technique.name, 'parameters'])])
            except KeyError:
                try:
                    technique.parameters.update(
                        self.project.idea.configuration['_'.join(
                            [technique.technique, 'parameters'])])
                except AttributeError:
                    pass
        return technique

    def _publish_selected(self, technique: 'Technique') -> 'Technique':
        """Limits parameters to those appropriate to the 'technique' 'technique'.

        If 'technique.techniquens.selected' is True, the keys from
        'technique.outline.defaults' are used to select the final returned
        parameters.

        If 'technique.outline.selected' is a list of parameter keys, then only
        those parameters are selected for the final returned parameters.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            'Technique': instance with modifications made.

        """
        if technique.outline.selected:
            if isinstance(technique.outline.selected, list):
                parameters_to_use = technique.outline.selected
            else:
                parameters_to_use = list(technique.outline.default.keys())
            new_parameters = {}
            for key, value in technique.parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            technique.parameters = new_parameters
        return technique

    def _publish_required(self, technique: 'Technique') -> 'Technique':
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            'Technique': instance with modifications made.

        """
        try:
            technique.parameters.update(technique.outline.required)
        except TypeError:
            pass
        return technique

    def _publish_search(self, technique: 'Technique') -> 'Technique':
        """Separates variables with multiple options to search parameters.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            'Technique': instance with modifications made.

        """
        technique.space = {}
        if technique.outline.hyperparameter_search:
            new_parameters = {}
            for parameter, values in technique.parameters.items():
                if isinstance(values, list):
                    if any(isinstance(i, float) for i in values):
                        technique.space.update(
                            {parameter: uniform(values[0], values[1])})
                    elif any(isinstance(i, int) for i in values):
                        technique.space.update(
                            {parameter: randint(values[0], values[1])})
                else:
                    new_parameters.update({parameter: values})
            technique.parameters = new_parameters
        return technique

    def _publish_runtime(self, technique: 'Technique') -> 'Technique':
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            'Technique': instance with modifications made.

        """
        try:
            for key, value in technique.outline.runtime.items():
                try:
                    technique.parameters.update({key: getattr(technique, value)})
                except AttributeError:
                    raise AttributeError(' '.join(
                        ['no matching runtime parameter', key, 'found']))
        except (AttributeError, TypeError):
            pass
        return technique

    def _publish_conditional(self, book: 'Book', technique: 'Technique') -> 'Technique':
        """Modifies 'parameters' based upon various conditions.

        A related class should have its own '_publish_conditional' method for
        this method to modify 'published'. That method should have a
        'parameters' and 'name' (str) argument and return the modified
        'parameters'.

        Args:
            technique ('Technique'): Technique instance to be modified.

        Returns:
            'Technique': instance with modifications made.

        """
        if 'conditional' in technique.outline:
            try:
                technique.parameters = book._publish_conditional(
                    name = technique.name,
                    parameters = technique.parameters)
            except AttributeError:
                pass
        return technique

    """ Core siMpLify Methods """

    def publish(self, step: str, technique: str) -> 'Technique':
        """Creates parameters and algorithms for a 'Technique' instance.

        Args:
            step (str): name of step in a 'Book' instance iterable.
            technique (str): name of the specific technique to use for a 'step'
                 in a 'Book' instance iterable.

        Returns:
            'Technique': instance with 'algorithm', 'parameters', and
                'data_dependent' attributes added.

        """

        if technique in ['none']:
            return None
        else:
            instance = Technique(name = step, technique = technique)
            outline = self.options[step][technique]
            instance.algorithm = outline.load('algorithm')
            instance.parameters = self._publish_parameters(outline = outline)
            instance.data_dependent = outline.data_dependent
        return instance


@dataclass
class Contributor(SimpleEditor):
    """Creates Chapter instances for a Book instance.

    Args:
        project ('Project'): a related Project instance.
        worker (str): name of the key to the Worker instance in the 'workers'
            attribute of 'project' for which this Class should draft a Book
            instance.

    """
    project: 'Project'
    worker: str

    def _publish_chapters(self) -> 'Book':
        """Publishes 'chapters' for a Book instance.

        Returns:
            Book instance.

        """
        book = self.project.library[self.worker]
        techniques = self.project.workers[self.worker].techniques
        options = self.project.workers[self.worker].options
        for step, techniques in techniques.items():
            for technique in techniques:
                algorithm = technique.load()
                parameters = self.parametizer.apply(
                    outline = options[technique])

            techniques = self._get_techniques(step = step)
            for technique in techniques:
                self.project.library[step]['techniques'][technique] = (
                    options[step][technique])

            parameters = self._publish_parameters(worker = worker)

        for i, step in enumerate(book.steps):
            if not step in book.techniques:
                book.techniques[step] = {}
            for technique in listify(book.techniques[i]):
                if not technique in ['none', None]:
                    technique = (options[step][technique].load())
                    book.contents[step][technique] = Technique(
                        book = book,
                        name = step,
                        technique = technique,
                        outline = options[step][technique])
                    book.contents[step][technique].algorithm = (
                        book.outline.load())
        return book

    def _publish_chapter_metadata(self,
            book: 'Book',
            number: int) -> Dict[str, Any]:
        """Finalizes metadata for Chapter instance.

        Args:
            book ('Book'): Book instance to be modified.
            number (int): chapter number; used for recordkeeping.

        Returns:
            Dict[str, Any]: metadata dict.

        """
        metadata = {'number': number + 1}
        try:
            metadata.update(book.metadata)
        except AttributeError:
            pass
        return metadata



@dataclass
class Scholar(SimpleEditor):
    """Base class for applying SimpleManuscript subclass instances to data.

    Args:
        project ('Project'): a related Project instance.
        worker (str): name of the key to the Worker instance in the 'workers'
            attribute of 'project' for which this Class should apply a Book
            instance.

    """
    project: 'Project'
    worker: str

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        return self

    """ Core siMpLify Methods """

    def apply(self,
            book: 'Book',
            data: Optional[Union['Ingredients', 'Book']] = None,
            **kwargs) -> Union['Ingredients', 'Book']:
        """Applies objects in 'manuscript' to 'data'.

        Args:
            book ('Book'): Book instance with algorithms to apply to 'data'.
            data (Optional[Union['Ingredients', 'Book']]): a data source for
                the 'book' methods to be applied.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's options' 'apply' method.

        Returns:
            Union['Ingredients', 'Book']: data object with modifications
                possibly made.

        """
        for chapter in book:
            for step, technique in chapter:
                data = technique.apply(data = data, **kwargs)
        return book

@dataclass
class Scholar(object):
    """Applies methods to siMpLify class instances.

    Args:
        project ('Project'): a related director class instance.

    """
    project: 'Project'

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Sets initial index location for iterable.
        self._position = 0
        return self

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

    def _apply_technique(technique: )

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

    def apply(self,
            book: 'Book',
            data: Optional[Union['Ingredients', 'Book']] = None,
            **kwargs) -> Union['Ingredients', 'Book']:
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
        for chapter in book:
            for step, technique in chapter:
                data = technique.apply(data = data,**kwargs)
        return book


    """ Core siMpLify Methods """

    def _add_data_dependents(self, data: object) -> None:
        """Completes parameter dictionary by adding data dependent parameters.

        Args:
            data (object): data object with attributes for data dependent
                parameters to be added.

        Returns:
            parameters with any data dependent parameters added.

        """
        if self.outline.data_dependents is not None:
            for key, value in self.outline.data_dependents.items():
                try:
                    self.parameters.update({key, getattr(data, value)})
                except KeyError:
                    print('no matching parameter found for', key, 'in',
                        data.name)
        return self

    def _add_parameters_to_algorithm(self) -> None:
        """Attaches 'parameters' to the 'algorithm'."""
        try:
            self.algorithm = self.algorithm(**self.parameters)
        except AttributeError:
            try:
                self.algorithm = self.algorithm(self.parameters)
            except AttributeError:
                pass
        except TypeError:
            pass
        return self

    """ Public Methods """

    def draft(self):
        """Creates 'algorithm' and 'outline' attributes."""
        # Injects attributes from Idea instance, if values exist.
        self = self.workers.idea.apply(instance = self)
        self.outline = self.workers[self.technique]
        self.algorithm = self.outline.load()
        return self

    def publish(self) -> None:
        """Finalizes 'algorithm' and 'parameters' attributes."""
        self.algorithm = self.algorithm.publish()
        self.parameters = self.parameters.publish()
        return self

    def apply(self, data: object, **kwargs) -> object:
        """

        """
        self._add_data_dependent(data = data)
        self._add_parameters_to_algorithm()
        try:
            self.algorithm.fit(
                getattr(data, ''.join(['x_', data.state])),
                getattr(data, ''.join(['y_', data.state])))
            setattr(
                data, ''.join(['x_', data.state]),
                self.algorithm.transform(getattr(
                    data, ''.join(['x_', data.state]))))
        except AttributeError:
            try:
                data = self.algorithm.apply(data = data)
            except AttributeError:
                pass
        return data

    """ Scikit-Learn Compatibility Methods """

    @DataValidator
    def fit(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional[object] = None) -> None:
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Raises:
            AttributeError if no 'fit' method exists for local 'algorithm'.

        """
        if x is not None:
            try:
                if y is None:
                    self.algorithm.process.fit(x)
                else:
                    self.algorithm.process.fit(x, y)
            except AttributeError:
                error = ' '.join([self.design.name,
                                  'algorithm has no fit method'])
                raise AttributeError(error)
        elif data is not None:
            self.algorithm.process.fit(
                getattr(data, ''.join(['x_', data.state])),
                getattr(data, ''.join(['y_', data.state])))
        else:
            error = ' '.join([self.name, 'algorithm has no fit method'])
            raise AttributeError(error)
        return self

    @DataValidator
    def fit_transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional[object] = None) -> (
                Union[pd.DataFrame, 'Ingredients']):
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or ingredients is not passed to
                the method.

        """
        self.algorithm.process.fit(x = x, y = y, data = ingredients)
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            return self.algorithm.process.transform(x = x, y = y)
        elif data is not None:
            return self.algorithm.process.transform(data = ingredients)
        else:
            error = ' '.join([self.name,
                              'algorithm has no fit_transform method'])
            raise TypeError(error)

    @DataValidator
    def transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional[object] = None) -> (
                Union[pd.DataFrame, 'Ingredients']):
        """Generic transform method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'process'.

        """
        if hasattr(self.algorithm.process, 'transform'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    return self.algorithm.process.transform(x)
                else:
                    return self.algorithm.process.transform(x, y)
            elif data is not None:
                return self.algorithm.process.transform(
                    X = getattr(data, 'x_' + data.state),
                    Y = getattr(data, 'y_' + data.state))
        else:
            error = ' '.join([self.name, 'algorithm has no transform method'])
            raise AttributeError(error)



# @dataclass
# class Book(SimpleCatalog):
#     """Stores and iterates Chapters.

#     Args:
#         project ('Project'): current associated project.

#     Args:
#         project ('Project'): associated Project instance.
#         options (Optional[Dict[str, 'Worker']]): SimpleCatalog instance or
#             a SimpleCatalog-compatible dictionary. Defaults to an empty
#             dictionary.
#         steps (Optional[Union[List[str], str]]): steps of key(s) to iterate in
#             'options'. Also, if not reset by the user, 'steps' is used if the
#             'default' property is accessed. Defaults to an empty list.

#     """
#     project: 'Project' = None
#     options: Optional[Dict[str, 'Worker']] = field(default_factory = dict)
#     steps: Optional[Union['SimpleSequence', List[str], str]] = field(
#         default_factory = list)
#     name: Optional[str] = None
#     chapter_type: Optional['Chapter'] = None
#     iterable: Optional[str] = 'chapters'
#     metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
#     file_format: Optional[str] = 'pickle'
#     export_folder: Optional[str] = 'book'

#     def __post_init__(self) -> None:
#         """Initializes class instance attributes."""
#         # Sets default 'name' attribute if none exists.
#         if self.name is None:
#             self.name = self.__class__.__name__.lower()
#         # Calls parent method for initialization.
#         super().__post_init__()
#         return self

#     """ Core SiMpLify Methods """

#     def apply(self,
#             options: Optional[Union[List[str], Dict[str, Any], str]] = None,
#             data: Optional[Union['Ingredients', 'Book']] = None,
#             **kwargs) -> Union['Ingredients', 'Book']:
#         """Calls 'apply' method for published option matching 'step'.

#         Args:
#             options (Optional[Union[List[str], Dict[str, Any], str]]): ordered
#                 options to be applied. If none are passed, the 'published' keys
#                 are used. Defaults to None
#             data (Optional[Union['Ingredients', 'Book']]): a siMpLify object for
#                 the corresponding 'options' to apply. Defaults to None.
#             kwargs: any additional parameters to pass to the options' 'apply'
#                 method.

#         Returns:
#             Union['Ingredients', 'Book'] is returned if data is passed;
#                 otherwise nothing is returned.

#         """
#         if isinstance(options, dict):
#             options = list(options.keys())
#         elif options is None:
#             options = self.default
#         self._change_active(new_active = 'applied')
#         for option in options:
#             if data is None:
#                 getattr(self, self.active)[option].apply(**kwargs)
#             else:
#                 data = getattr(self, self.active)[option].apply(
#                     data = data,
#                     **kwargs)
#             getattr(self, self.active)[option] = getattr(
#                 self, self.active)[option]
#         if data is None:
#             return self
#         else:
#             return data


# @dataclass
# class Chapter(SimpleCatalog):
#     """Iterator for a siMpLify process.

#     Args:
#         book ('Book'): current associated Book
#         metadata (Optional[Dict[str, Any]], optional): any metadata about the
#             Chapter. Unless a subclass replaces it, 'number' is automatically a
#             key created for 'metadata' to allow for better recordkeeping.
#             Defaults to an empty dictionary.

#     """
#     book: 'Book' = None
#     name: Optional[str] = None
#     iterable: Optional[str] = 'book.steps'
#     metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
#     file_format: Optional[str] = 'pickle'
#     export_folder: Optional[str] = 'chapter'

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     """ Private Methods """

#     def _apply_extra_processing(self) -> None:
#         """Extra actions to take."""
#         return self

#     """ Core siMpLify Methods """

#     def apply(self, data: Optional['Ingredients'] = None, **kwargs) -> None:
#         """Applies stored 'options' to passed 'data'.

#         Args:
#             data (Optional[Union['Ingredients', 'SimpleManuscript']]): a
#                 siMpLify object for the corresponding 'step' to apply. Defaults
#                 to None.
#             kwargs: any additional parameters to pass to the step's 'apply'
#                 method.

#         """
#         if data is not None:
#             self.ingredients = data
#         for step in getattr(self, self.iterable):
#             self.book[step].apply(data = self.ingredients, **kwargs)
#             self._apply_extra_processing()
#         return self