from setuptools import setup
from setuptools import find_packages

setup(name='forkan',
      version='0.0.1',
      description='Deep Learning experiments',
      author='Luca Lach',
      author_email='llach@teachfak.uni-bielefeld.de',
      url='https://github.com/llach/dl-toolkit',
      install_requires=['tensorflow-gpu>=1.11.0',
                        'tensorflow >= 1.11.0',
                        'tabulate>=0.8.2',
                        'coloredlogs>=10.0',
                        'matplotlib>=3.0.0',
                        'keras>=2.2.4'],
      packages=find_packages())
