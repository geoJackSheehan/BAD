#!/usr/bin/env bash

tests=(
    test_fad.py
    test_elementary_functions.py
    test_derivs.py
)

export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

# decide what driver to use (depending on arguments given)
if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    driver="${@} -m unittest"
elif [[ $# -gt 0 && ${1} == 'pytest' ]]; then
    driver="${@}"
elif [[ $# -gt 0 && ${1} == 'CI' ]]; then
    # Assumes the package has been installed and dependencies resolved.  This
    # would be the situation for a customer.  Uses `pytest` for testing.
    shift
    unset PYTHONPATH
    driver="pytest ${@}"
else
    driver="python ${@} -m unittest"
fi

# run the tests
${driver} ${tests[@]}
