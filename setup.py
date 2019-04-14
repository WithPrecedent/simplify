from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='ml_funnel',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="ml_funnel offers flexible, accessible tools for parsing and analysis of court opinions",
    author="Corey Rayburn Yung",
    author_email='coreyyung@ku.edu',
    url='https://github.com/with_precedent/ml_funnel',
    packages=['ml_funnel'],
    entry_points={
        'console_scripts': [
            'ml_funnel=ml_funnel.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='ml_funnel',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ]
)
