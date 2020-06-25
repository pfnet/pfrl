#!/bin/bash

set -Ceu

gpu="$1"

for SCRIPT in $(find examples_tests/ -type f -name '*.sh')
do
  echo "Running example tests: ${SCRIPT}"
  bash ${SCRIPT} ${gpu}
done
