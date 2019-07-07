from setuptools import setup, find_namespace_packages

# TODO read dependencies from requirement file
# read description from readme.md

setup(
    name='congressional_adversary',
    description='work in progress',
    version='0.2',
    python_requires='>=3.7.0',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    author='Albert Webson',
    author_email='awebson@cs.brown.edu',
    license='MIT'
)
