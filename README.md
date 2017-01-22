# Convection in axially symmetric accretion discs with microscopic transport coefficients

This repository presents the pipelines described in the paper Malanchev, Postnov and Shakura (2017) (further [MPS2017](http://adsabs.harvard.edu/abs/2017MNRAS.464..410M)).
The code is written on Python 3 with usage of scipy and matplotlib.
Full list of requirements is listed in [`radiative_cond_disc/requirements.txt`](https://github.com/hombit/convinstab/blob/master/radiative_cond_disc/requirements.txt).


## Code

### Optically thin discs with electron heat conductivity
Jupyter notebook [`heat_cond_disc.ipynb`](https://github.com/hombit/convinstab/blob/master/heat_cond_disc.ipynb) corresponds to Section 2 of [MPS2017](http://adsabs.harvard.edu/abs/2017MNRAS.464..410M).
The notebook provides three figures:m
  - Vertical distribution of dimensionless temperature *`θ(x)`*. See Fig. 1 from [MPS2017](http://adsabs.harvard.edu/abs/2017MNRAS.464..410M).
  - Size of the laminar zone for different Prandtl numbers `Pr`. The parameters of conductivity *`a`* and viscosity *`b`* are fixed.
  - Dependence of thickness parameter *`ξ`* on Prandtl number `Pr` for constant *`a`* and *`b`*.
  
### Radiative heat conductivity
Folder [`radiative_cond_disc`](https://github.com/hombit/convinstab/tree/master/radiative_cond_disc) corresponds to Section 3 of [MPS2017](http://adsabs.harvard.edu/abs/2017MNRAS.464..410M).

#### `radiative_cond_disc/vertstr.py`
File [`vertstr.py`](https://github.com/hombit/convinstab/blob/master/radiative_cond_disc/vertstr.py) contains class `FindPi` that solves system of ODEs similar to [Ketsaris and Shakura (1998)](http://adsabs.harvard.edu/abs/1998A%26AT...15..193K).
This file can be used as script to obtain Pi1, Pi2, Pi3, Pi4. See an example:
```shell
$ python3 radiative_cond_disc/vertstr.py 1e2
Pi1 = 4.98451
Pi2 = 0.57649
Pi3 = 1.12619
Pi4 = 0.39527
```
Use `--help` to read the script documentation.

Moreover this file can be used as a module.
The following code performs the same calculations as the example above (alpha-disc with Kramer's opacity law and value of free paramter *`τ0`* equals 100):
```python
from vertstr import FindPi
fp = FindPi(100)
Pi = fp.getPi()
print(Pi)
```
The code of `vertstr` is documented with [docstrings](https://www.python.org/dev/peps/pep-0257/), so you can view documentation in your terminal or web-browser using [pydoc](https://docs.python.org/3/library/pydoc.html):
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
Module [`plot`](https://github.com/hombit/convinstab/blob/master/radiative_cond_disc/plot.py) contains functions used to plot vertical distributions of dimensionless variables (see Fig. 2 and 3 in [MPS2017](http://adsabs.harvard.edu/abs/2017MNRAS.464..410M)).

#### `radiative_cond_disc/dlogTdlogP_b.ipynb`
Jupyter notebook [`dlogTdlogP_b.ipynb`](https://github.com/hombit/convinstab/blob/master/radiative_cond_disc/dlogTdlogP_b.ipynb) contains a figure that shows the dependence of symmetry plane value of *`d log(T) / d log(P)`* on the viscosity parameter *`b`* in the case of ion heat conductivity *`(d = 0)`*.


## Licence
Copyright (c) 2016, Konstantin L. Malanchev

Please, accompany any results obtained using this code with reference to Malanchev, Postnov and Shakura (2017) [2017MNRAS.464..410M](http://adsabs.harvard.edu/abs/2017MNRAS.464..410M)


## BibTex
```bibtex
@ARTICLE{2017MNRAS.464..410M,
   author = {{Malanchev}, K.~L. and {Postnov}, K.~A. and {Shakura}, N.~I.
	},
    title = "{Convection in axially symmetric accretion discs with microscopic transport coefficients}",
  journal = {\mnras},
archivePrefix = "arXiv",
   eprint = {1609.03799},
 primaryClass = "astro-ph.HE",
 keywords = {accretion, accretion discs, convection},
     year = 2017,
    month = jan,
   volume = 464,
    pages = {410-417},
      doi = {10.1093/mnras/stw2348},
   adsurl = {http://adsabs.harvard.edu/abs/2017MNRAS.464..410M},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
