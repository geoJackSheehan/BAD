[![.github/workflows/coverage.yml](https://code.harvard.edu/CS107/team23/actions/workflows/coverage.yml/badge.svg?branch=milestone2)](https://code.harvard.edu/CS107/team23/actions/workflows/coverage.yml)
[![.github/workflows/test.yml](https://code.harvard.edu/CS107/team23/actions/workflows/test.yml/badge.svg?branch=milestone2)](https://code.harvard.edu/CS107/team23/actions/workflows/test.yml)


# Getting Started

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
>>> from bad_package.ad_interface import *
>>> from bad_package.elementary_functions import *
>>> from bad_package.fad import *
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
