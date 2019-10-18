"""
.. module:: animate
:synopsis: animated data visualizations
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import ArtistTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {}


@dataclass
class Animate(SimpleIterable):
    """Creates animated data visualizations.

    Args:
        steps(dict(str: ArtistTechnique)): names and related ArtistTechnique classes for
            creating data visualizations.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_draft (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_publish (bool): whether to call the 'implement' method when the class
            is instanced.
    """
    steps: object = None
    name: str = 'animator'
    auto_draft: bool = True
    auto_publish: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self