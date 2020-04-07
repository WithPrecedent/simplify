"""
.. module:: system
:synopsis: project workflow made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections.abc
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


@dataclasses.dataclass
class SimpleSystem(collections.abc.Iterable):
    """Base class for siMpLify project stages.

    A 'SimpleSystem' subclass maintains a progress state stored in the attribute
    'stage'. The 'stage' corresponds to whether one of the core workflow
    methods has been called. The string stored in 'stage' can then be used by
    subclasses to alter instance behavior, call methods, or change access
    method functionality.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        stages (Optional[Union[List[str], Dict[str, str]]]): list of recognized
            states which correspond to methods within a class instance or a dict
            with keys of recognized states and values for different iterables to
            return when the '__iter__' method is called. Defaults to
            ['initialize', 'draft', 'publish', 'apply'].
        auto_advance (Optional[bool]): whether to automatically advance 'stage'
            when one of the stage methods is called (True) or whether 'stage'
            must be changed manually by using the 'advance' method. Defaults to
            True.

    """
    name: Optional[str] = None
    stages: Optional[Union[List[str], Dict[str, str]]] = dataclasses.field(
        default_factory = lambda: ['initialize', 'draft', 'publish', 'apply'])
    auto_advance: Optional[bool] = True

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Sets 'name' to the default value if it is not passed.
        self.name = self.name or self.__class__.__name__.lower()
        # Validates that corresponding methods exist for each stage after the
        # first stage and constructs attributes for the class iterable.
        self._validate_stages()
        # Sets initial stage.
        self.stage = self.stages[0]
        # Automatically calls stage methods if attribute named 'auto_{stage}'
        # is set to True.
        self._auto_stages()
        return self

    """ Public Methods """

    def advance(self, stage: Optional[str] = None) -> None:
        """Advances to next stage in 'stages' or to 'stage' argument.

        This method only needs to be called manually if 'auto_advance' is False.
        Otherwise, this method is automatically called when individual stage
        methods are called via '__getattribute__'.

        If this method is called at the last stage, it does not raise an
        IndexError. It simply leaves 'stage' at the last item in the list.

        Args:
            stage(Optional[str]): name of stage matching a string in 'stages'.
                Defaults to None. If not passed, the method goes to the next
                'stage' in stages.

        Raises:
            ValueError: if 'stage' is neither None nor in 'stages'.

        """
        self.previous_stage = self.stage
        if stage is None:
            try:
                self.stage = self.stages[self.stages.index(self.stage) + 1]
            except IndexError:
                pass
        elif stage in self.stages:
            self.stage = stage
        else:
            raise ValueError(f'{stage} is not a recognized stage')
        return self

    """ Required ABC Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable for class instance, depending upon 'stage'.

        Returns:
            Iterable: different depending upon stage.

        """
        try:
            return iter(getattr(self, self._iterables[self.stage]))
        except AttributeError:
            return iter(self.stages)

    """ Dunder Methods """

    def __getattribute__(self, attribute: str) -> Any:
        """Changes 'stage' if one of the corresponding methods are called.

        If attribute matches any item in 'stages', the 'stage' attribute is
        assigned to 'attribute.'

        Args:
            attribute (str): name of attribute sought.

        """
        if self.auto_advance:
            try:
                if attribute in super().__getattribute__('stages'):
                    super().__getattribute__('advance')(stage = attribute)
            except AttributeError:
                pass
        return super().__getattribute__(attribute)

    """ Private Methods """

    def _validate_stages(self) -> None:
        """Validates 'stages' type and existence of corresponding methods.

        Raises:
            AttributeError: if a method listed in 'stages' does not exist.

        """
        # Converts 'stages' to a list and stores dict in '_iterables'.
        if isinstance(self.stages, dict):
            self._iterables = copy(self.stages)
            self.stages = list(self.stages.keys())
        # Tests whether stage methods listed in 'stages' exist.
        for stage in self.stages:
            if stage not in dir(instance) and stage not in [self.stages[0]]:
                raise AttributeError(f'{stage} is not in {self.name}')
        return self

    def _auto_stages(self) -> None:
        """Calls stage method if corresponding boolean is True."""
        # Automatically calls stage methods if attribute named 'auto_{stage}'
        # is set to True.
        for stage in self.stages[1:]:
            try:
                if getattr(self, f'auto_{stage}'):
                    getattr(self, stage)()
            except AttributeError:
                pass
        return self