"""
.. module:: plan
:synopsis: iterable container
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.base import SimpleComposite


@dataclass
class SimplePlan(object):
    """Iterator for a siMpLify process.

    Args:
        steps (Optional[Union[List[str], str]]): names of techniques to be
            applied. These names should match keys in the 'options' attribute.
            If using the Idea instance settings, this argument should not be
            passed. Default is None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    steps: Dict[str, SimpleComposite]
    name: Optional[str] = 'simple_plan'
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.research()
        self.draft()
        return self

    """ Dunder Methods """

    def __add__(self, steps: Dict[str, SimpleComposite]) -> None:
        """Adds step(s) at the end of 'steps'.

        Args:
            steps (Dict[str: SimpleComposite]): the next step(s) to be added.

        """
        self.add(steps = steps)
        return self

    def __iadd__(self, steps: Dict[str, SimpleComposite]) -> None:
        """Adds step(s) at the end of 'steps'.

        Args:
            steps (Dict[str: SimpleComposite]): the next step(s) to be added.

        """
        self.add(steps = steps)
        return self

    """ Import/Export Methods """

    def load(self,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None) -> None:
        """Loads 'steps' from disc.

        For any arguments not passed, default values stored in the shared Depot
        instance will be used based upon the current 'stage' of the siMpLify
        project.

        Args:
            file_path (str): a complete file path for the file to be loaded.
            folder (str): a path to the folder where the file should be loaded
                from (not used if file_path is passed).
            file_name (str): contains the name of the file to be loaded without
                the file extension (not used if file_path is passed).

        """
        self.steps = self.depot.load(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = 'pickle')
        return self

    def save(self,
            file_path: Optional[str] = None,
            folder: Optional[str]  = None,
            file_name: Optional[str]  = None) -> None:
        """Exports 'steps' to disc.

        For any arguments not passed, default values stored in the shared Depot
        instance will be used based upon the current 'stage' of the siMpLify
        project.

        Args:
            file_path (str): a complete file path for the file to be saved.
            folder (str): a path to the folder where the file should be saved
                (not used if file_path is passed).
            file_name (str): contains the name of the file to be saved without
                the file extension (not used if file_path is passed).

        """
        self.depot.save(
            variable = self.steps,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = 'pickle')
        return self

    """ Steps Methods """

    def add(self, steps: Dict[str, SimpleComposite]) -> None:
        """Adds step(s) at the end of 'steps'.

        Args:
            steps: Dict[str: SimpleComposite]): the next step(s) to be added.

        """
        try:
            self.steps.update(steps)
        except TypeError:
            self.steps.update(steps.steps)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        return self

    def publish(self, ingredients: SimpleComposite, **kwargs) -> None:
        """Applies 'steps' to passed 'ingredients'.

        Args:
            data ('SimpleComposite'): a data container or other SimpleComposite
                for steps to be applied to.

        """
        setattr(self, ingredients.name, ingredients)
        for step, technique in self.steps.items():
            try:
                self.stage = step
            except KeyError:
                pass
            setattr(self, ingredients.name,
                    technique.publish(ingredients = ingredients, **kwargs))
        return self

