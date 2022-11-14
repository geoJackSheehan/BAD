#!/usr/bin/env bash
# Test harness to properly discover source code modules

tests = (
    test_fad.py

)

export PYTHONPATH = "$(pwd -P)/../src":${PYTHONPATH}

pytest