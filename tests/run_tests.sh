#!/usr/bin/env bash
# Test harness to properly discover source code modules

export PYTHONPATH = "$(pwd -P)/../src":${PYTHONPATH}

pytest