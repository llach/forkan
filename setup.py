from setuptools import setup
from setuptools import find_packages

setup(name='forkan',
      version='0.0.1',
      description='Deep Learning experiments',
      author='Luca Lach',
      author_email='llach@teachfak.uni-bielefeld.de',
      url='https://github.com/llach/dl-toolkit',
      install_requires=['tensorflow-gpu>=1.11.0',
                        'numpy>=1.15.3',
                        'tabulate>=0.8.2',
                        'Pillow>=5.3.0',
                        'gym>=0.10.9',
                        'coloredlogs>=10.0',
                        'matplotlib>=3.0.0',
                        'keras>=2.2.4',
                        'baselines'],
      packages=find_packages())
