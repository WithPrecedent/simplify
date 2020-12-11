"""
resources: default settings, options, and rules
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    settings (dict): default settings for a Settings instance.

    
"""
from __future__ import annotations
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import simplify
import sourdough 


raw_algorithms: sourdough.types.Catalog[str, simplify.SimpleTechnique] = (
    sourdough.types.Catalog(contents = { 
        'eli5': simplify.SimpleTechnique(
            name = 'eli5',
            module = 'simplify.critic.algorithms',
            algorithm = 'Eli5Explain'),
        'shap': simplify.SimpleTechnique(
            name = 'shap',
            module = 'simplify.critic.algorithms',
            algorithm = 'ShapExplain'),
        'simplify': simplify.SimpleTechnique(
            name = 'shap',
            module = 'simplify.critic.algorithms',
            algorithm = 'SimplifyExplain'),
        'skater': simplify.SimpleTechnique(
            name = 'skater',
            module = 'simplify.critic.algorithms',
            algorithm = 'SkaterExplain'),
        'sklearn': simplify.SimpleTechnique(
            name = 'sklearn',
            module = 'simplify.critic.algorithms',
            algorithm = 'SklearnExplain')}))


def get_algorithms(settings: Mapping[str, Any]) -> sourdough.types.Catalog:
    """[summary]

    Args:
        project (sourdough.Project): [description]

    Returns:
        sourdough.types.Catalog: [description]
        
    """
    return algorithms  