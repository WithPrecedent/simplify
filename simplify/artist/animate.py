"""
.. module:: animate
:synopsis: animated data visualizations
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.plan import SimplePlan


@dataclass
class Animate(SimplePlan):
    """Creates animated data visualizations.

    Args:
        steps(dict(str: SimpleStep)): names and related SimpleStep classes for
            creating data visualizations.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_implement (bool): whether to call the 'implement' method when the class
            is instanced.
    """
    steps: object = None
    name: str = 'animator'
    auto_publish: bool = True
    auto_implement: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        return self