#!/usr/bin/env bash

export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

pytest --cov=bad_package --cov-fail-under=90