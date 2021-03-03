import codecs
from setuptools import find_packages
from setuptools import setup

install_requires = [
    'torch>=1.3.0',
    'gym>=0.9.7',
    'numpy>=1.10.4',
    'pillow',
    'filelock',
]

test_requires = [
    'pytest',
    'attrs<19.2.0',  # pytest does not run with attrs==19.2.0 (https://github.com/pytest-dev/pytest/issues/3280)  # NOQA
]

setup(name='pfrl',
      version='0.2.1',
      description='PFRL, a deep reinforcement learning library',
      long_description=codecs.open('README.md', 'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      author='Yasuhiro Fujita',
      author_email='fujita@preferred.jp',
      license='MIT License',
      packages=find_packages(),
      install_requires=install_requires,
      test_requires=test_requires)
