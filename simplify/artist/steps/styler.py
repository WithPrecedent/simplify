"""
.. module:: styler
:synopsis: sets universal data visualization style
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns

from simplify.core.definitionsetter import SimpleDirector
from simplify.core.definitionsetter import Option


@dataclasses.dataclass
class Styler(SimpleIterable):
    """Sets data visualization style.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    name: str = 'styler'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        """Sets fonts, colors, and styles for plots that do not have set styles.
        """
        # List of colorblind colors obtained from here:
        # https://www.dataquest.io/blog/making-538-plots/.
        # Thanks to Alex Olteanu.
        colorblind_colors = [[0,0,0], [230/255,159/255,0],
                             [86/255,180/255,233/255], [0,158/255,115/255],
                             [213/255,94/255,0], [0,114/255,178/255]]
        sns.set_style(style = self.seaborn_style)
        sns.set_context(context = self.seaborn_context)
        plt.style.use(style = self.plot_style)
        plt.rcParams['font.family'] = self.plot_font
        if self.seaborn_palette == 'colorblind':
            sns.set_palette(color_codes = colorblind_colors)
        else:
            sns.set_palette(palette = self.seaborn_palette)
        return self