"""
.. module:: handler
:synopsis: project application made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import abc
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import pathos.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

from simplify.core import component


@dataclasses.dataclass
class SimpleHandler(abc.ABC):
    """Base class for creating or modifying other siMpLify classes."""

    """ Required Subclass Methods """

    @abc.abstractmethod
    def apply(self, 
            data: component.SimpleComponent, **kwargs) -> component.SimpleComponent:
        """Subclasses must provide their own methods."""
        return self


@dataclasses.dataclass
class SimpleParallel(SimpleHandler):
    """Applies workflow using one or more CPU or GPU cores.

    Args:
        gpu (Optional[bool]): whether to use GPU cores, when possible, to
            parallelize operations (True) or to solely use CPU cores (False).
            Defaults to False.

    """

    gpu: Optional[bool] = False

    """ Private Methods """

    def _apply_gpu(self, process: Callable, data: object, **kwargs) -> object:
        """

        """
        try:
            return process(data, **kwargs)
        except TypeError:
            return self._apply_cpu(process = process, data = data, **kwargs)

    def _apply_cpu(self, process: Callable, data: object, **kwargs) -> object:
        """

        """
        results = []
        arguments = data
        arguments.update(kwargs)
        with mp.Pool() as pool:
            results.append(pool.starmap(method, arguments))
        pool.close()
        return results

    """ Core siMpLify Methods """

    def apply(self, process: Callable, data: object, **kwargs) -> object:
        """

        """
        if self.gpu:
            return self._apply_gpu(process = process, data = data, **kwargs)
        else:
            return self._apply_cpu(process = process, data = data, **kwargs)