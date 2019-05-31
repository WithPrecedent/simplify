from setuptools import setup, find_packages

__version__ = '0.1.0'

setup(name = 'siMpLify',
      description = "Making machine learning simple",
      author = "Corey Rayburn Yung",
      author_email = 'coreyrayburnyung@gmail.com',
      url = 'https://github.com/with_precedent/simplify',
      packages = find_packages(exclude = ['tests*']),
      include_package_data = True,
      version = __version__,
      entry_points = {'console_scripts': ['simplify = simplify.cli:cli']},
      python_requires = '>= 3.6',
      install_requires = open('requirements.txt').readlines(),
      keywords = 'data science machine learning pandas sklearn',
      classifiers = ['Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7'])
