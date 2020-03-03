"""
.. module:: wisconsin breast cancer classification
:synopsis: example using sklearn breast cancer data
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""
import os
import sys
sys.path.insert(0, os.path.join('..', 'simplify'))
sys.path.insert(0, os.path.join('..', '..', 'simplify'))

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from simplify import Project


# Loads cancer data and converts from numpy arrays to a pandas DataFrame.
cancer = load_breast_cancer()
df = pd.DataFrame(
    data = np.c_[cancer['data'], cancer['target']],
    columns = np.append(cancer['feature_names'], ['target']))
# Sets root_folder for data and results exports.
# root_folder = os.path.join('..', '..')
# Sets location of configuration settings for the project. Depending upon your
# OS and python configuration, one of these might work better.
idea = Path.cwd().joinpath('examples', 'cancer_settings.ini')
#idea = os.path.join(os.getcwd(), 'cancer_settings.ini')

# Creates siMpLify project, automatically configuring the process based upon
# settings in the 'idea_file'.
cancer_project = Project(
    idea = idea,
    # filer = root_folder,
    dataset = df)
# Converts label to boolean type to correct numpy default above.
cancer_project.dataset.change_datatype(
    columns = 'target',
    datatype = 'boolean')
# Fills missing data with appropriate default values based on column datatype.
# cancer_project.dataset.smart_fill()
# Iterates through every recipe and exports plots, explainers, and other
# metrics from each recipe.
cancer_project.apply()
# Outputs information about the best recipe to the terminal.
# cancer_project['critic'].print_best()
# Saves dataset file with predictions or predicted probabilities added
# (based on options in idea).
#cancer_project.dataset.save(file_name = 'cancer_df')