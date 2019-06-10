import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from simplify import Codex, Settings
from simplify.cookbook import Cookbook

# Loads cancer data and converts from numpy arrays to pandas dataframe.
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns = np.append(cancer['feature_names'], ['target']))
# Loads settings file.
settings = Settings(file_path = 'cancer_settings.ini')
# Creates instance of Data with the cancer codexframe.
codex = Codex(settings = settings, df = df)
# Converts label to boolean type - conversion from numpy arrays leaves all
# columns as float type.
codex.change_column_type(columns = ['target'], data_type = bool)
# Fills missing codex with appropriate default values based on column type.
codex.smart_fillna()
# Creates instance of Cookbook.
cookbook = Cookbook(codex = codex, settings = settings)
# Automatically creates list of recipes cookbook.recipes based upon settings
# file.
cookbook.prepare()
# Iterates through every recipe and exports plots from each recipe.
cookbook.bake()
# Creates and exports a table of summary statistics from the codexframe.
codex.summarize()
# Saves the recipes, results, and cookbook.
cookbook.save_everything()
# Outputs information about the best recipe.
cookbook.print_best()
# Saves codex file.
codex.save(file_name = 'cancer_df')