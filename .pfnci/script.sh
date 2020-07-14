#!/bin/bash
# script.sh is a script to run Docker for testing.  This is called by CI like
# "bash .pfnci/script.sh py3.cpu".  If a machine running the script has no
# GPUs, this should fall back to CPU testing automatically.  This script
# requires that a corresponding Docker image is accessible from the machine.
# TODO(imos): Enable external contributors to test this script on their
# machines.  Specifically, locate a Dockerfile generating chainer-ci-prep.*.
#
# Usage: .pfnci/script.sh [target]
# - target is a test target (e.g., "py3.cpu").
#
# Environment variables:
# - GPU (default: 0) ... Set a number of GPUs to GPU.  GPU=0 disables GPU
#       testing.
# - DRYRUN ... Set DRYRUN=1 for local testing.  This disables destructive
#       actions and make the script print commands.
# - XPYTEST ... Set XPYTEST=/path/to/xpytest-linux for testing xpytest.  It will
#       replace xpytest installed inside a Docker image with the given binary.
#       It should be useful to test xpytest.

set -eu

cd "$(dirname "${BASH_SOURCE}")"/..

################################################################################
# Main function
################################################################################
main() {
  TARGET="$1"

  # Initialization.
  prepare_docker &
  prepare_xpytest &
  wait

  # Prepare docker args.
  docker_args=(docker run  --rm --volume="$(pwd):/src:ro" --volume="/root/.pfrl:/root/.pfrl/") 
  if [ "${GPU:-0}" != '0' ]; then
    docker_args+=(--ipc=host --privileged --env="GPU=${GPU}" --runtime=nvidia)
  fi
  if [ "${XPYTEST:-}" == '' ]; then
    docker_args+=(--volume="$(pwd)/bin/xpytest:/usr/local/bin/xpytest:ro")
  else
    docker_args+=(--volume="${XPYTEST}:/usr/local/bin/xpytest:ro")
  fi
  docker_args+=(--env="XPYTEST_NUM_THREADS=${XPYTEST_NUM_THREADS:-$(nproc)}")
  docker_args+=(--env="TEST_EXAMPLES=${TEST_EXAMPLES:-0}")

  # Determine base image to use.
  docker_image=pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
  docker_args+=(--env="SLOW=${SLOW:-0}")

  for ZIP in a3c_results.zip dqn_results.zip iqn_results.zip rainbow_results.zip ddpg_results.zip trpo_results.zip ppo_results.zip td3_results.zip sac_results.zip
  do
      gsutil cp gs://chainerrl-asia-pfn-public-ci/${ZIP} .
      mkdir -p ~/.pfrl/models/
      unzip ${ZIP} -d ~/.pfrl/models/
      rm ${ZIP}
  done

  run "${docker_args[@]}" "${docker_image}" bash /src/.pfnci/run.sh "${TARGET}"
}

################################################################################
# Utility functions
################################################################################

# run executes a command.  If DRYRUN is enabled, run just prints the command.
run() {
  echo '+' "$@"
  if [ "${DRYRUN:-}" == '' ]; then
    "$@"
  fi
}

# prepare_docker makes docker use tmpfs to speed up.
# CAVEAT: Do not use docker during this is running.
prepare_docker() {
  # Mount tmpfs to docker's root directory to speed up.
  if [ "${CI:-}" != '' ]; then
    run service docker stop
    run mount -t tmpfs -o size=100% tmpfs /var/lib/docker
    run service docker start
  fi
  # Configure docker to pull images from gcr.io.
  run gcloud auth configure-docker
}

# prepare_xpytest prepares xpytest.
prepare_xpytest() {
  run mkdir -p bin
  run curl -L https://github.com/chainer/xpytest/releases/download/v0.1.0/xpytest-linux --output bin/xpytest
  run chmod +x bin/xpytest
}

################################################################################
# Bootstrap
################################################################################
main "$@"
