from setuptools import setup
from setuptools import find_packages

setup(name='dl-toolkit',
      version='0.0.1',
      description='Deep Learning experiments',
      author='Luca Lach',
      author_email='llach@teachfak.uni-bielefeld.de',
      url='https://github.com/llach/dl-toolkit',
      install_requires=['tensorflow>=1.11.0',
                        'matplotlib>=3.0.0'],
      packages=find_packages())
