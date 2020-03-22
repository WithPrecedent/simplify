"""
.. module:: idea test
:synopsis: tests Idea class
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from pathlib import pathlib.Path
from simplify.core.idea import Idea

def test_idea():
    ini_idea = Idea(configuration = 'idea_settings.ini', infer_types = True)
    assert ini_idea.configuration == {
        'general': {'verbose': True, 'seed': 43},
        'simplify': {'simplify_steps': ['analyst', 'critic']}}
    # py_idea = Idea(configuration = 'idea_settings.py', infer_types = True)
    # assert py_idea.configuration == {
    #     'general': {'verbose': True, 'seed': 43},
    #     'simplify': {'simplify_steps': ['analyst', 'critic']}}
    # csv_idea = Idea(configuration = 'idea_settings.csv', infer_types = True)
    # assert csv_idea.configuration == {
    #     'general': {'verbose': True, 'seed': 43},
    #     'simplify': {'simplify_steps': ['analyst', 'critic']}}
    return


if __name__ == '__main__':
    test_idea()