[![.github/workflows/coverage.yml](https://code.harvard.edu/CS107/team23/actions/workflows/coverage.yml/badge.svg?branch=main)](https://code.harvard.edu/CS107/team23/actions/workflows/coverage.yml)
[![.github/workflows/test.yml](https://code.harvard.edu/CS107/team23/actions/workflows/test.yml/badge.svg?branch=main)](https://code.harvard.edu/CS107/team23/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# CS107 Final Project - Team 23

# Team Members:

Feel free to contact the team via email for any further questions:

Annabel Yim ([annabelyim@g.harvard.edu](mailto:user@example.com))

Hope Neveux ([hopeneveux@g.harvard.edu](mailto:user@example.com))

Jack Sheehan ([jacksheehan@g.harvard.edu](mailto:user@example.com))

Danhee Kim ([sharonkim@g.harvard.edu](mailto:user@example.com))

# Documentation

Documentation for this package can be found [here](docs/documentation.ipynb)

# Getting Started

## Installation via PyPi

1. Ensure your Python version is 3.8 or greater

2. Ensure `pip` and `setuptools` are installed and up to date

    ```
    # Linux (Ubuntu), MacOS
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip

    python -m pip install setuptools
    ```

3. This package depends on `numpy` version 1.21.0 and greater. Please ensure it is installed and up to date.

    ```
    # Linux (Ubuntu), MacOS
    python -m pip install numpy
    python -m pip install --upgrade numpy
    ```

4. Pip install from the test PyPI

    ```
    python -m pip install https://test.pypi.org/simple/ bad_package
    ```

## Installation via GitHub

1. Clone the package repository to folder. Make sure to set up a SSH key beforehand.  

    ```
    mkdir bad_package
    cd bad_package
    git clone git@code.harvard.edu:CS107/team23.git
    cd team23
    ```

2. Install virtualenv on your machine if not already installed.

    ```
    pip install virtualenv
    ```

3. Create virtual environment

    ```
    virtualenv cs107
    ```

4. Activate the new virutal environment

    Mac OS or Linux:

    ```python
    source cs107/bin/activate
    ```

    Windows:

    ```python
    cs107\Scripts\activate    
    ```

5. Install package and its requirements

    ```python
    pip install ./
    ```    

6. To deactivate virtual environment

    ```python
    deactivate
    ```        

## Using Forward Mode

### Import modules

```python
>>> from bad_package.interface import *
>>> from bad_package.elementary_functions import *
>>> from bad_package.fad import *
>>> from bad_pacakge.rad import *
>>> import numpy as np
```

### How to use Forward Mode

#### Scalar

```python
# User defines the function that they want to optimize.
>>> def scalar(x):
>>>     return 4*x + 3

# User creates a 1D numpy array of initial value for input to the function they want to optimize
>>> x = np.array([2])

# User instantiate AutoDiff class
>>> ad = AutoDiff(scalar, x)

# User can call the primal trace and jacobian matrix
>>> ad.compute()
>>> print(f'Primal: {ad.get_primal()}')
Primal: 11
>>> print(f'Tangent: {ad.get_jacobian()}')
Tangent: [4]
```

#### Vector

```python
# User defines the function that they want to optimize.
>>> def vector(x):
>>>     return x[0]**2 + 3*x[1] + 5

# User creates a N-D numpy array of initial value for input to the function they want to optimize
>>> x = np.array([1, 2])

# User instantiate AutoDiff class
>>> ad = AutoDiff(vector, x)

# User can call the primal trace and jacobian matrix
>>> ad.compute()
>>> print(f'Primal: {ad.get_primal()}')
Primal: 12
>>> print(f'Tangent: {ad.get_jacobian()}')
Tangent: [2, 3]
```

## Using Reverse Mode

### How to use Reverse Mode

#### Scalar

```python
# User defines the function that they want to optimize.
>>> def scalar(x):
>>>     return 4*x + 3

# User creates a 1D numpy array/float/scalar of initial value for input to the function they want to optimize
>>> x = np.array([2])

# User instantiate ReverseAD class
>>> rm = ReverseAD(scalar, x)

# User can call the jacobian matrix
>>> print(f'Jacobian: {rm.get_jacobian()}')
Jacobian: [4]
```

#### Vector

```python
# User defines the function that they want to optimize.
>>> def vector(x):
>>>     return x[0]**2 + 3*x[1] + 5

# User creates a N-D numpy array of initial value for input to the function they want to optimize
>>> x = np.array([1, 2])

# User instantiate ReverseAD class
>>> rm = ReverseAD(vector, x)

# User can call the jacobian matrix
>>> print(f'Jacobian: {rm.get_jacobian()}')
Jacobian: [2, 3]
```

#### Vector 2


```python
# User defines the functions that they want to optimize.
>>> def vector1(x):
>>>     return (5*x + 50)/(2*x**2)

>>> def vector2(x):
>>>     return 10 + 2*x

>>> func = np.array([vector1, vector2])

# User creates a 1D numpy array/float/scalar of initial value for input to the function they want to optimize
>>> x = np.array([5])

# User instantiate ReverseAD class
>>> rm = ReverseAD(vector, x)

# User can call the jacobian matrix
>>> print(f'Jacobian: {rm.get_jacobian()}')
Jacobian: [-0.5, 2]
```

# Broader Impact and Inclusivity Statement

## Broader Impact

As developers of an open-sourced Auto Differentiation package, we care about bringing diversity, equity, and inclusion to the open-source community. We hope that our package can be used by a diverse group of people, including women, people of color, people with disabilities, and an exhaustive list consisting of minorities. We hope we can provide a safe space for motivating others to contribute and point out possible changes that need to be made for this open-sourced package.

This open-sourced package may be misused and cause some serious ethical issues. Although this package is an efficient tool to solve complex gradient problems without solving them by hand, prior to using this package, we hope users can spend time understanding the real mathematical implications and uses of taking derivatives and gradients. Our package is primarily used for academic purposes and should not be for sale.

## Software Inclusivity

Team23 is dedicated to creating the software space more inclusive to underrepresented groups. To promote this standard, the BAD package will be released to the open source community welcoming contribution from any person. The core developers of BAD package believe that any individual, regardless of background, is capable of making meaningful contributions within the software community. We welcome _all_ developers that believe in fostering a respectful environment to contributing to our package. To make BAD package more accessible to non-native English speakers, we have plans to translate our documentation to various languages. We welcome any new ideas and feedback to improve the BAD package! Team23 believes that the best work is a product of collaboration of diverse backgrounds.

Pull requests will be reviewed blindly among the core developers to mitigate bias. If not approved by a majority vote, relevant supportive feedback will be provided.

Any unethical activity under the BAD package that fails to uphold this inclusive standard will not be tolerated. Discrimination is not welcome here.