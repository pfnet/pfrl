#!/bin/bash

set -eux

# Use latest black to apply https://github.com/psf/black/issues/1288
pip3 install git+git://github.com/psf/black.git@88d12f88a97e5e4c8fd0d245df0a311e932fd1e1 flake8 mypy isort

black --diff --check pfrl tests examples
isort --diff --check pfrl tests examples
flake8 pfrl tests examples
mypy pfrl
# mypy does not search child directories unless there is __init__.py
find tests -type f -name "*.py" | xargs dirname | sort | uniq | xargs mypy
# To avoid "duplicate module named xxx.py" errors, run mypy independently for each directory
find examples -type f -name "*.py" | xargs dirname | sort | uniq | xargs -I{} mypy {}
