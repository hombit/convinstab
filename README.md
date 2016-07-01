# Convection in axially symmetric accretion discs with microscopic transport coefficients

This repo presents pipelines described in the paper Malanchev, Postnov and Shakura (2016) (further MPS2016).
All code is written on Python 3 with usage of scipy and matplotlib.
Full list of requirements is listed in [`radiative_cond_disc/requirements.txt`](https://github.com/hombit/convinstab/blob/master/radiative_cond_disc/requirements.txt).


## Code

### Optically thin discs with electron heat conductivity
Jupyter notebook [`heat_cond_disc.ipynb`](https://github.com/hombit/convinstab/blob/master/heat_cond_disc.ipynb) is corresponding to Section 2 of MSP2016.
The notebook provides three figures:
  - Vertical distribution of dimensionless temperature `θ(x)`. This is Fig. 1 from MPS2016.
  - Size of the laminar zone for different Prandtl numbers `Pr` and constant `a` and `b`.
  - Dependence of thickness parameter `ξ` on Prandtl number `Pr` for constant `a` and `b`.
  
### Radiative heat conductivity
Folder [`radiative_cond_disc`](https://github.com/hombit/convinstab/tree/master/radiative_cond_disc) is corresponding to Section 3 of MPS2016.

#### `radiative_cond_disc/vertstr.py`
File [`vertstr.py`](https://github.com/hombit/convinstab/blob/master/radiative_cond_disc/vertstr.py) contains class `FindPi` that solves system of ODEs similar to Ketsaris and Shakura (1998).
This file can be used as script to obtain Pi1, Pi2, Pi3 and Pi4 values:
```shell
$ python3 radiative_cond_disc/vertstr.py 1e2
Pi1 = 4.98451
Pi2 = 0.57649
Pi3 = 1.12619
Pi4 = 0.39527
```
Read script documentation with `--help` argument

On other hand this file can be used as module.
This code does the same calculations as above example of script usage (alpha-disc with Kramer's opacity law and value of free paramter `τ0` equals 100):
```python
from vertstr import FindPi
fp = FindPi(100)
Pi = fp.getPi()
```
The code of `vertstr` is documented with [docstrings](https://www.python.org/dev/peps/pep-0257/), you can view documentation in your terminal or web-browser using [pydoc](https://docs.python.org/3/library/pydoc.html):
```shell
$ pydoc3 vertstr
Help on module vertstr:

NAME
    vertstr

CLASSES
    builtins.object
        FindPi
...
```

#### `radiative_cond_disc/plot.py`
Module [`plot`](https://github.com/hombit/convinstab/blob/master/radiative_cond_disc/plot.py) contains functions used to plot vertical distributions of dimensionless variables (Fig. 2 and 3 in MPS2016).

#### `radiative_cond_disc/dlogTdlogP_b.ipynb`
Jupyter notebook [`dlogTdlogP_b.ipynb`](https://github.com/hombit/convinstab/blob/master/radiative_cond_disc/dlogTdlogP_b.ipynb) contains figure that shows dependence of symmetry plane value of `d log(T) / d log(P)` on parameter `b` in the case of `d = 0`.


## Licence
Copyright (c) 2016, Konstantin L. Malanchev

Please accompany any results obtained using this code with reference to Malanchev, Postnov and Shakura (2016), in prep.
