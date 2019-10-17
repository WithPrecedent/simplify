"""
.. module:: style
:synopsis: sets universal data visualization style
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

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
class Style(SimpleIterable):

    technique: object = None
    parameters: object = None
    name: str = 'style'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self


    def _set_style(self):
        """Sets fonts, colors, and styles for plots that do not have set
        styles.
        """
        # List of colorblind colors obtained from here:
        # https://www.dataquest.io/blog/making-538-plots/.
        # Thanks to Alex Olteanu.
        colorblind_colors = [[0,0,0], [230/255,159/255,0],
                             [86/255,180/255,233/255], [0,158/255,115/255],
                             [213/255,94/255,0], [0,114/255,178/255]]
        plt.style.use(style = self.plot_style)
        plt.rcParams['font.family'] = self.plot_font
        sns.set_style(style = self.seaborn_style)
        sns.set_context(context = self.seaborn_context)
        if self.seaborn_palette == 'colorblind':
            sns.set_palette(color_codes = colorblind_colors)
        else:
            sns.set_palette(palette = self.seaborn_palette)
        return self