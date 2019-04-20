from setuptools import setup

setup(name = 'ml_funnel',
      description = "ml_funnel is a high-level pipeline and grid creation tool for machine learning",
      author = "Corey Rayburn Yung",
      author_email = 'coreyyung@ku.edu',
      url = 'https://github.com/with_precedent/ml_funnel',
      packages = ['ml_funnel'],
      entry_points = {'console_scripts': ['ml_funnel = ml_funnel.cli:cli']},
      python_requires = '>= 3.6',
      install_requires = open('requirements.txt').readlines(),
      keywords = 'ml_funnel data science machine learning pandas sklearn',
      classifiers = ['Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7'])
