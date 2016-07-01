#!/usr/bin/env python3


import argparse
import numpy as np
from copy import copy
from enum import IntEnum
from numbers import Integral, Real
from scipy import optimize, integrate, interpolate


class _TestAction(argparse._StoreTrueAction):
    '''
    argparse Action for test running
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        import doctest
        print('Running doctest...')
        doctest.testmod()
        print('Test is OK')
        parser.exit()


class Vars(IntEnum):
    '''
    Enumerate that contains names of unknown functions.
    All functions are dimensionless, for more info look KS1998.
    
    Attributes
    ----------
    p
        Pressure
    z
        Vertical coordinate, equals 0 in the plane of symmetry of the disc
    q
        Flux of energy
    t
        Temperature
    '''
    p = 0
    z = 1
    q = 2
    t = 3


# Number of items in Vars enumerate
lenVars = len(Vars.__members__)


def entropy_tp(t, p):
    '''
    Dimensionless entropy from temperate and pressure.
    Output array is normalized on its maximum absolute value

    Parameters
    ----------
    t : array_like
        Temperature.
    p : array_like
        Pressure.

    Returns
    -------
    array
    '''
    s = 2.5 * np.log(t) - np.log(p)
    s /= np.max( np.abs(s) )
    return s

def dlogT_dlogP(t, p):
    '''
    Logarithmic derivate of temperature of pressure

    Parameters
    ----------
    t : array_like
        Temperature.
    p : array_like
        Pressure.

    Returns
    -------
    array_like
    '''
    logp = np.log(p)
    tck = interpolate.splrep( logp, np.log(t), k=1 )
    dlogdlog = interpolate.splev( logp, tck, der=1 )
    return dlogdlog


class FindPi(object):
    __Pi = None
    __Pi0 = np.array( [5, 0.5, 1, 0.4] )
    _Pi0 = __Pi0

    __doc__ = '''
    Solver of the system of dimensionless vertical structure ODEs similar to
    the system in Ketsaris, Shakura (1998) (KS1998). The system contains four
    linear differential equations for four unknown variables: pressure p,
    vertical coordinate z, flux of energy q and temperature t. The only
    argument is dimensionless mass coordinate sigma. The system contains eight
    first-type boundary conditions (two for each function) and four unknown
    parameters `Pi`. These unknown parameters depend on one free parameter
    `tau`. The result of the solution is values of this four parameters Pi
    (method `getPi`) and distributions of four unknown functions (method
    `pzqt`).

    Parameters
    ----------
    tau : positive float, optional
        Free parameter of the problem. It corresponds to tau_0 for
        ``absorption`` transfer and delta for ``scattering`` transfer from
        KS1998.
    Pi0 : array_like, optional
        Initial guess. Default is typical values from KS1998:
        ``{Pi0}``
    reverse : bool, optional
        Specifies direction of ODEs integration. If False than integrate from
        `sigma` = 0 to `sigma` = 1 (from plane of symmetry to photosphere), if
        True than in opposite direction. True is default and usually shows
        better divergence of optimization process.
    heating : str or pair of floats, optional
        Heating law: dq/dz ~ t^b * p^d. Should be ``(b, d)`` pair
        or one of following strings:
        
            - 'alpha' describes Shakura-Syunyaev alpha-disc, dq/dz ~ p,
              dq/dsigma ~ t, b = 0, d = 1.
            - 'const' describes constant energy generation per unit mass,
              dq/dz ~ p/t, dq/dsigma ~ 1, b = -1, d = 1.
            - 'ion' describes energy release by microscopic ion viscosity,
              dq/dz ~ t^2.5, dq/dsigma ~ t^3.5/p, b = 2.5, d = 0.
        
        Default is ``alpha``.
    transfer : str, optional
        Type of energy transfer in the disc. It sets type of free parameter tau
        (tau_0 or delta from KS1998), type of boundary conditions for t(1)
        and p(1) and default opacity law (see ``opacity`` description bellow).
        Should be one of
        
            - 'absorption' absorption dominates over scattering.
              Default ``opacity`` law is varkappa ~ rho/t^3.5, varsigma = 1,
              psi = 3.5 (Kramer\'s opacity law).
            - 'scattering' scattering dominates over absorption.
              Default ``opacity`` law is varkappa ~ 1, varsigma = 0, psi = 0
              (Thomson scattering).
         
        Default is ``absorption``.
    opacity : pair of floats or None, optional
        ``(varsigma, psi)`` pair that describes opacity law:
        varkappa ~ rho^varsigma / t^psi.
        If None then it is setted by ``transfer`` parameter (see its
        description above).

    Attributes
    ----------
    Pi : array or None
        The main result of calculations. None for fresh object
    b : float
        From heating law ``eta ~ t^b p^d``
    d : float
        From heating law ``eta ~ t^b p^d``
    varsigma : float
        From opacity law ``varkappa ~ rho^varsigma / t^psi``
    psi : float
        From opacity law ``varkappa ~ rho^varsigma / t^psi``
    _Pi0 : array
        Default value of initial guess for Pi0. It should equals
        ``{Pi0}``

    Methods
    -------
    getPi()
        Solve optimization problem and returns array with parameters Pi.
        Raise RuntimeError if optimization failed.
    dlogTdlogP_centr()
        Value of d log T / d log P in the plane of symmetry of the disc.
    pzqt(sigma=1000)
        Distribution of unknown functions on `sigma`.

    Notes
    -----
    Optimization problem is solved with the same bounds (0.1, 10) for all of
    four parameters Pi.

    Examples
    --------
    Find Pi values for alpha-disc and Kramer's opacities and tau_0 = 100:

    >>> import numpy as np
    >>> fp = FindPi(100)
    >>> Pi = fp.getPi()
    >>> print( np.round(Pi, decimals=3) )
    [ 4.985  0.576  1.126  0.395]
    
    For some parameters we have to set initial guess Pi0:
    >>> fp = FindPi(
    ...     10,
    ...     Pi0=[2, 0.7, 1, 0.6],
    ...     heating='ion',
    ...     transfer='scattering'
    ... )
    >>> Pi = fp.getPi()
    >>> print( np.round(Pi, decimals=3) )
    [ 2.632  0.737  0.996  0.418]
    '''.format(Pi0=__Pi0)

    def __init__(self, 
        tau,
        Pi0 = None,
        reverse = True,
        heating = 'alpha',
        transfer = 'absorption',
        opacity = None
    ):
        self.__tau = tau
        self.__reverse = reverse
        self.__heating = heating
        self.__transfer = transfer

        if Pi0 is not None:
            self.__Pi0 = Pi0

        if heating == 'alpha':
            self.__b = 0.
            self.__d = 1.
        elif heating == 'const':
            self.__b = -1.
            self.__d = 1.
        elif heating == 'ion':
            self.__b = 2.5
            self.__d = 0.
        elif len(heating) == 2 and isinstance(heating[0], Real) and isinstance(heating[1], Real):
            self.__b, self.__d = self.heating
            self.__heating = 'custom'
        else:
            raise ValueError('Unknown heating type {}'.format(heating))

        if opacity is None:
            if transfer == 'absorption':
                self.__opacity = (1., 3.5)
            elif transfer == 'scattering':
                self.__opacity = (0., 0.)
            else:
                raise ValueError('Unknown transfer type {}'.format(transfer))
        elif len(opacity) == 2 and isinstance(opacity[0], Real) and isinstance(opacity[1], Real):
            self.__opacity = opacity
        else:
            raise ValueError('Unknown opacity type {}. It should be either None or sequence of two real numbers'.format(opacity))

        self.__varsigma, self.__psi = self.__opacity

        self.__log10tau = np.log10(self.__tau)

        self.__y0 = np.empty(lenVars)
        self.__y0[Vars.p] = 1.
        self.__y0[Vars.z] = 0.
        self.__y0[Vars.q] = 0.
        self.__y0[Vars.t] = 1.

        if transfer == 'absorption':
            self.__f_tau2over3 = integrate.quad(
                lambda x: (1. + 1.5*x)**((self.__psi+self.__varsigma)/4.),
                0., 2./3.
            )[0]

        if self.__reverse:
            self.__sigma = np.array([1., 0.])
        else:
            self.__sigma = np.array([0., 1.])

    @property
    def tau(self):
        return self.__tau

    @property
    def Pi0(self):
        return copy(self.__Pi0)

    @property
    def reverse(self):
        return self.__reverse

    @property
    def heating(self):
        return self.__heating

    @property
    def b(self):
        return self.__b

    @property
    def d(self):
        return self.__d

    @property
    def transfer(self):
        return self.__transfer

    @property
    def opacity(self):
        return self.__opacity

    @property
    def varsigma(self):
        'From opacity law varkappa ~ rho^varsigma / t^psi'
        return self.__varsigma

    @property
    def psi(self):
        'From opacity law varkappa ~ rho^varsigma / t^psi'
        return self.__psi

    @property
    def y0(self):
        'Left boundary conditions'
        return copy(self.__y0)

    @property
    def Pi(self):
        'Result of optimization'
        return copy(self.__Pi)

    def _derivatives(self, y, sigma, Pi):
        '''
        Right side of ODEs. See KS1998 for heating 'alpha' or 'const'

        Parameters
        ----------
        y : array-like
            Current values of unknown function, indexes are described in `Vars`
        sigma : float
            Current value of sigma
        Pi : array-like

        Returns
        -------
        array
        '''
        dy = np.empty(lenVars)
        dy[Vars.p] = -Pi[0] * Pi[1] * y[Vars.z]
        dy[Vars.z] = Pi[1] * y[Vars.t] / y[Vars.p]
        dy[Vars.q] = Pi[2] * y[Vars.t]**(self.__b+1) * y[Vars.p]**(self.__d-1)
        dy[Vars.t] = -Pi[3] * y[Vars.q] * y[Vars.p]**(self.__varsigma) / y[Vars.t]**(self.__psi+self.__varsigma+3)
        return dy

    def _y1(self, Pi):
        '''
        Right boundary conditions. See KS1998

        Parameters
        ----------
        Pi : array-like

        Returns
        -------
        array
        '''
        y1 = np.empty(lenVars)
        
        y1[Vars.z] = 1
        y1[Vars.q] = 1

        if self.__transfer == 'absorption':
            y1[Vars.t] = ( 16./3. * Pi[3] / self.__tau )**0.25
            y1[Vars.p] = (
                    3. * (self.__varsigma+1.) / (16. * 2.**((self.__psi+self.__varsigma)/4.) )
                    * Pi[0] * Pi[1] / Pi[3]
                    * y1[Vars.t]**(self.__psi+self.__varsigma+4.)
                    * self.__f_tau2over3
                )**(1./(self.__varsigma+1))
        elif self.__transfer == 'scattering':
            y1[Vars.t] = ( 4 * Pi[3] / self.__tau )**0.25
            y1[Vars.p] = Pi[0] * Pi[1] / self.__tau
        else:
            raise ValueError('Unknown transfer type {}'.format(self.__transfer))
        
        return y1

    def _integrate(self, Pi, sigma=None):
        '''
        Integrate ODEs and returns values at pints described in `sigma`

        Parameters
        ----------
        Pi : array-like
        sigma : array-like or None, optional
            Values of sigma points for which to find `y`. The first point
            should be 0 for `reverse` == False and 1 for `reverse` == True.
            If None than use `__sigma` that is ``[1,0]`` for `reverse` == True
            and ``[0,1]`` otherwise.

        Returns
        -------
        array
            Array containing the value of `y` for each point described in
            `sigma`.
        '''
        if sigma is None:
            sigma = self.__sigma
        
        if self.__reverse:
            y_init = self._y1(Pi)
        else:
            y_init = self.__y0

        ys = integrate.odeint(
            self._derivatives,
            y_init,
            sigma,
            args=(Pi,),
        )
        return ys

    def _discrepancy(self, Pi):
        '''
        Discrepancy between values of `y` described in boundary condition and
        received from `_integrate`.
        '''

        if self.__reverse:
            return np.sum( (self._integrate(Pi)[-1] - self.__y0)**2 ) / lenVars
        else:
            return np.sum( (self._integrate(Pi)[-1] - self._y1(Pi))**2 ) / lenVars

    def getPi(self):
        '''
        Solve optimization problem and returns array with parameters Pi.
        Raise RuntimeError if optimization failed.

        Returns
        -------
        array
            Values of unknown parameters Pi
        '''
        if self.__Pi is not None:
            return self.__Pi

        opt_res = optimize.minimize(
            self._discrepancy,
            self.__Pi0,
            bounds = ((0.1, 10,),) * lenVars
        )
        
        if opt_res.status == 0:
            self.__Pi = opt_res.x
            return self.__Pi
        else:
            raise RuntimeError('Cannon solve optimization problem. Corresponding OptimizeResult is\n{}'.format(opt_res))

    def dlogTdlogP_centr(self):
        '''
        Return value of d log T / d log P at the plane of symmetry of the disc.
        '''
        Pi = self.getPi()
        return Pi[2] * Pi[3] / ( Pi[0] * Pi[1]**2 )

    def pzqt(self, sigma=1000):
        '''
        Distribution of unknown functions on `sigma`.

        Parameters
        ----------
        sigma : array or int, optional
            If type is ``array`` then `sigma` should be monotonous sequence of
            sigma points with the first element equals 1 if `reverse` == True
            and 0 otherwise.
            If type is ``int`` then linear mesh with `sigma` points between 0
            and 1 is used.

        Returns
        -------
        array, shape(n)
            Array with `sigma` values
        array, shape(n, len(Pi))
            Array with corresponding values of unknown functions
        '''
        if isinstance(sigma, Integral):
            if self.__reverse:
                sigma = np.linspace(1, 0, sigma)
            else:
                sigma = np.linspace(0, 1, sigma)
        else:
            sigma = np.asarray(sigma)

        ys = self._integrate(self.getPi(), sigma=sigma)
        return sigma, ys


###################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute four Pi values')
    parser.add_argument(
        '-T', '--test',
        dest = 'test',
        action = _TestAction,
        help = 'run tests and exit',
    )
    parser.add_argument(
        'tau',
        action = 'store',
        type = float,
        default = 1e6,
        help = 'value of free parameter (tau0 for --transfer=absorption or delta for --transfer=scattering)',
    )
    parser.add_argument(
        '-e', '--heating',
        dest = 'heating',
        choices = ['alpha', 'const', 'ion'],
        default = 'alpha',
        help = 'type of energy release',
    )
    parser.add_argument(
        '-c', '--transfer',
        dest = 'transfer',
        choices = ['absorption', 'scattering'],
        default = 'absorption',
        help = 'main process in radiative conductivity',
    )
    parser.add_argument(
        '-o', '--opacity',
        dest = 'opacity',
        action = 'store',
        nargs = 2,
        type = float,
        metavar = ('VARSIGMA', 'PSI'),
        help = "parameters of opacity law: varkappa ~ rho^varsigma / t^psi, default is Kramer's law (1 3.5) for --transfer=absorption and Thomson scattering (0 0) for --transfer=scattering",
    )
    args = parser.parse_args()

    fp = FindPi(
        args.tau,
        heating = args.heating,
        transfer = args.transfer,
        opacity = args.opacity
    )
    try:
        Pi = fp.getPi()
        print( '\n'.join( 'Pi{} = {:.5f}'.format(i+1, p) for i, p in enumerate(Pi) ) )
    except RuntimeError:
        print("Sorry, Pi hasn't calculated. Try another arguments")
