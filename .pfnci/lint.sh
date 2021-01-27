#!/bin/bash

set -eux

pip3 install black flake8 'mypy>=0.800' isort

black --diff --check pfrl tests examples
isort --diff --check pfrl tests examples
flake8 pfrl tests examples
mypy pfrl tests
# To avoid "duplicate module named xxx.py" errors, run mypy independently for each directory
find examples -type f -name "*.py" | xargs dirname | sort | uniq | xargs -I{} mypy {}
