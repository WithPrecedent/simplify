"""
.. module:: defaults
:synopsis: default values for core siMpLify classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from numpy import datetime64
from pandas.api.types import CategoricalDtype

"""
Default values for core classes are stored here, at the top of the module,
outside of Defaults, so that they can be easily found and edited for users
wanting to make changes.

Alternatively, users can use the 'add_defaults' method to replace any default
values after the Defaults class is instanced. If that option is chosen, the
dictionaries of defaults values should be referred to by the lower case versions
of the dictionary names (e.g., 'INGREDIENTS' should be 'ingredients').
"""

IDEA = {
    'infer_types': True,
    'override': False}

FILER = {
    'root_folder': ['..', '..'],
    'results_folder': 'results',
    'data_folder': 'data',
    'data_subfolders': ['raw', 'interim', 'processed', 'external'],
    'export_data_folders': {
            'sow': 'raw',
            'reap': 'raw',
            'clean': 'interim',
            'bale': 'interim',
            'deliver': 'interim',
            'chef': 'processed',
            'critic': 'processed'},
    'export_data_file_names': {
            'sow': None,
            'harvest': None,
            'clean': 'harvested_data',
            'bale': 'cleaned_data',
            'deliver': 'baled_data',
            'chef': 'final_data',
            'critic': 'final_data'},
    'export_data_file_formats': {
            'sow': 'source_format',
            'harvest': 'source_format',
            'clean': 'interim_format',
            'bale': 'interim_format',
            'deliver': 'interim_format',
            'chef': 'final_format',
            'critic': 'final_format'},
    'export_data_folders': {
            'sow': 'raw',
            'reap': 'interim',
            'clean': 'interim',
            'bale': 'interim',
            'deliver': 'processed',
            'chef': 'processed',
            'critic': 'recipe'},
    'export_data_file_names': {
            'sow': None,
            'harvest': 'harvested_data',
            'clean': 'cleaned_data',
            'bale': 'baled_data',
            'deliver': 'final_data',
            'chef': 'final_data',
            'critic': 'predicted_data'},
    'export_data_file_formats': {
            'sow': 'source_format',
            'harvest': 'interim_format',
            'clean': 'interim_format',
            'bale': 'interim_format',
            'deliver': 'final_format',
            'chef': 'final_format',
            'critic': 'final_format'},
    'override': False}

INGREDIENTS = {
    'default_df': 'df',
    'options': {
        'unsplit': {
            'train': 'full_suffix',
            'test': None},
        'train_test': {
            'train': 'train_suffix',
            'test': 'test_suffix'},
        'train_val': {
            'train': 'train_suffix',
            'test': 'test_suffix'},
        'full': {
            'train': 'full_suffix',
            'test': 'full_suffix'}},
    'data_prefixes': ['x', 'y'],
    'train_suffix': 'train',
    'test_suffix': 'test',
    'validation_suffix': 'val',
    'full_suffix': '',
    'datatypes': {
        'boolean': bool,
        'float': float,
        'integer': int,
        'string': object,
        'categorical': CategoricalDtype,
        'list': list,
        'datetime': datetime64,
        'timedelta': timedelta},
    'default_values': {
        'boolean': False,
        'float': 0.0,
        'integer': 0,
        'string': '',
        'categorical': '',
        'list': [],
        'datetime': 1/1/1900,
        'timedelta': 0}}


@dataclass
class Defaults(object):
    """Stores and injects default values into core siMpLify classes.

    The values here are outside of the Idea configuration structure because they
    either:
        1) are not something an average user needs to adjust;
        2) are of a datatype that cannot easily and intuitively be stored in
            all of the supported file formats used by Idea; and/or
        3) are mutable datatypes that can be passed to siMpLify objects, but
            the pythonic preference for immutable defaults is used in the
            objects.

    """
    def __post_init__(self):
        self.draft()
        return self

    """ Public Methods """

    def add_defaults(self, object_name: str, defaults: dict) -> None:
        """Adds new default values to be applied.

        Args:
            object_name (str): name of object for default values to be applied.
            defaults (dict): a dictionary containing attribute names for keys
                and attribute values for values.

        """
        if hasattr(self, object_name):
            getattr(self, object_name).update(defaults)
        else:
            setattr(self, object_name, defaults)
        return self

    """ Core siMpLify Methods """

    def draft(self):
        """Sets default values for core siMpLify objects."""
        for default in ('FILER', 'IDEA', 'INGREDIENTS'):
            self.add_defaults(
                object_name = default.lower(),
                defaults = globals()[default])
        return self

    def publish(self):
        """'publish' is unnecessary for this class."""
        return self

    def apply(self,
            instance: object,
            override: Optional[bool] = False) -> object:
        """Adds default values to passed instance.

        Args:
            instance (object): object for values to be added to attributes.
            override (Optional[bool]): whether to override the existing values
                of attributes, if not None. Defaults to False.

        Returns:
            object: with changed attriute values.

        """
        try:
            for attribute, value in getattr(self, instance.name).items():
                if (override
                        or not hasattr(instance, attribute)
                        or getattr(instance, attribute) is None):
                    setattr(instance, attribute, value)
        except AttributeError:
            print('No default values exist for', instance.name)
        return instance