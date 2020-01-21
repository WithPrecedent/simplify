"""
.. module:: siMpLify chef
:synopsis: siMpLify chef package
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

from simplify.chef.chef import Chef
from simplify.chef.chef import Cookbook
from simplify.chef.chef import Cookware


__all__ = [
    'Chef',
    'Cookbook',
    'Cookware']

COMPONENTS = {
    'scholar': 'Chef',
    'book': 'Cookbook',
    'catalog': 'Cookware'}