#!/usr/bin/env python3


import numpy as np
from copy import copy
from enum import IntEnum
from numbers import Real
from scipy import optimize, integrate, interpolate


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


def entropy(t, p):
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
    s /= np.abs( s.max() )
    return s

def dlogt_dlogp(t, p):
    '''
    Logarithmic derivate of temperature of pressure

    Parameters
    ----------
    t : array_like
        Temperature.
    p : array_like
        Pressure.

    Returns
    ------_
    array_like
    '''
    logp = np.log(p)
    tck = interpolate.splrep( logp, np.log(t), k=1 )
    dlogdlog = interpolate.splev( logp, tck, der=1 )
    return dlogdlog

def plot_vert_struct(fp, n=100, filename=None):
    '''
    Plot distributions of various functions to the EPS file.
    This function plots all unknown functions, normalized entropy `s` and
    derivative d log(t) / d log(p).

    Parameters
    ----------
    fp : FindPi
        Object used to calculate variables
    n : positive int, optional
        Number of points in sigma mesh.
    filename : str or None, optional
        Path of the filename to save plot. If None construct filename of
        the format ``{heating}_{transfer}_logtau_{logtau}.eps`` in the
        local directory.
    '''
    dashes = [
        (10000,),
        (6,6),
        (2,6),
        (8,4,2,4),
        (2,2,),
        (2,4,2,4,2,8)
    ]

    log10tau = np.log10(fp.tau)

    if filename is None:
        filename = '{}_{}_logtau_{}.pdf'.format(fp.heating, fp.transfer, log10tau)

    sigma, ys = fp.mesh(n)

    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('text', usetex=True)
    plt.title(
        r'Heating is \textit{{{heating}}}, transfer is \textit{{{transfer}}}, $\log{{\tau}} = {logtau:.1f}$\\{Pi}'.format(
            heating = fp.heating,
            transfer = fp.transfer,
            logtau = log10tau,
            Pi = ',  '.join( map(
                lambda i: '$\Pi_{} = {:.3f}$'.format(i+1, fp.Pi[i]),
                range(fp.Pi.shape[0])
            ) )
        ),
        multialignment = 'center'
    )
    plt.xlabel(r'$\sigma$')
    plt.ylim([0, 1])
    lines = []
    lines += plt.plot(sigma, ys[:,Vars.p], label=r'$p$')
    lines += plt.plot(sigma, ys[:,Vars.z], label=r'$z$')
    lines += plt.plot(sigma, ys[:,Vars.q], label=r'$q$')
    lines += plt.plot(sigma, ys[:,Vars.t], label=r'$t$')
    lines += plt.plot(
        sigma,
        entropy(ys[:,Vars.t],
        ys[:,Vars.p]),
        label=r'$s$'
    )
    lines += plt.plot(
        sigma,
        dlogt_dlogp(ys[:,Vars.t], ys[:,Vars.p]),
        label=r'$\frac{d \log{t}}{d \log{p}}$'
    )
    plt.setp(lines, color='k')
    for i, line in enumerate(lines):
#        plt.setp(line, linestyle=':')
        plt.setp(line, dashes=dashes[i])
    plt.legend(loc='best')
    plt.savefig(filename)


class FindPi(object):
    '''
    Solver of the system of dimensionless vertical structure ODEs similar to
    the system in Ketsaris, Shakura (1998) (KS1998). The system contains four
    linear differential equations for four unknown variables: pressure,
    vertical coordinate z, flux of energy and temperature. The only argument is
    dimensionless mass coordinate sigma. The system contains eight first-type
    boundary conditions (two for each function) and four unknown parameters
    `Pi`. These unknown parameters depend on one free parameter `tau`. The
    result of the solution is values of this four parameters Pi (method
    `getPi`) and distributions of four unknown functions (methods `mesh` and
    `plot_mesh`).

    Parameters
    ----------
    tau : positive float, optional
        Free parameter of the problem. It corresponds to tau_0 for ``ff``
        transfer and delta for ``thomson`` transfer from KS1998.
    Pi0 : array_like, optional
        Initial guess. Default is typical values from KS1998.
    reverse : bool, optional
        Specifies direction of ODEs integration. If False than integrate from
        `sigma` = 0 to `sigma` = 1 (from plane of symmetry to photosphere), if
        True than in opposite direction. True is default and usually shows
        better divergence of optimization process.
    heating : str or sequence, optional
        Heating law: dq/dz ~ t^b * p^d. Should be ``(b, d)`` pair
        or one of following strings:
        
            - 'alpha' describes Shakura-Syunyaev alpha-disc, dq/dz ~ p,
              dq/dsigma ~ t, b = 0, d = 1.
            - 'const' describes constant energy generation per unit mass,
              dq/dz ~ p/t, dq/dsigma ~ 1, b = -1, d = 1.
            - 'microvisc' describes heating by microscopic ion viscosity,
              dq/dz ~ t^2.5, dq/dsigma ~ t^3.5/p, b = 2.5, d = 0.
        
        Default is ``alpha``.
    transfer : str, optional
        Type of energy transfer in the disc. It sets type of free parameter tau
        (tau_0 or delta from KS1998), type of boundary conditions for t(1)
        and p(1) and default opacity law (see ``opacity`` description bellow).
        Should be one of
        
            - 'ff' absorption dominates over scattering.
              Default ``opacity`` law is kappa ~ rho/t^3.5, varsigma = 1,
              psi = 3.5 (Kramer\'s opacity law).
            - 'thompson' scattering dominates over absorption.
              Default ``opacity`` law is kappa ~ 1, varsigma = 0, psi = 0
              (Thomson scattering).
         
        Default is ``ff``.
    opacity : sequence or None, optional
        ``(varsigma, psi)`` pair that describes opacity law:
        kappa ~ rho^varsigma / t^psi.
        If None then it is setted by ``transfer`` parameter (see its
        description above).

    Attributes
    ----------
    Pi : array or None
        The main result of calculations. None for fresh object

    Methods
    -------
    getPi()
        Solve optimization problem and returns array with parameters Pi.
        Raise RuntimeError if optimization failed.
    mesh(n=1000)
        Distribution of unknown functions of the sigma mesh with n points

    Notes
    -----
    Optimization problem is solved with the same bounds (0.1, 10) for all of
    four parameters Pi.
    '''

    __Pi = None

    def __init__(self, 
        tau,
        Pi0 = np.array( [5, 0.5, 1, 0.4] ),
        reverse = True,
        heating = 'alpha',
        transfer = 'ff',
        opacity = None
    ):
        self.__tau = tau
        self.__Pi0 = Pi0
        self.__reverse = reverse
        self.__heating = heating
        self.__transfer = transfer

        if heating == 'alpha':
            self.__b = 0.
            self.__d = 1.
        elif heating == 'const':
            self.__b = -1.
            self.__d = 1.
        elif heating == 'microvisc':
            self.__b = 2.5
            self.__d = 0.
        elif len(heating) == 2 and isinstance(heating[0], Real) and isinstance(heating[1], Real):
            self.__b, self.__d = self.heating
            self.__heating = 'custom'
        else:
            raise ValueError('Unknown heating type {}'.format(heating))

        if opacity is None:
            if transfer == 'ff':
                self.__opacity = (1., 3.5)
            elif transfer == 'thompson':
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

        if transfer == 'ff':
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
        'From opacity law kappa ~ rho^varsigma / t^psi'
        return self.__varsigma

    @property
    def psi(self):
        'From opacity law kappa ~ rho^varsigma / t^psi'
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

        if self.__transfer == 'ff':
            y1[Vars.t] = ( 16./3. * Pi[3] / self.__tau )**0.25
            y1[Vars.p] = (
                    3. * (self.__varsigma+1.) / (16. * 2.**((self.__psi+self.__varsigma)/4.) )
                    * Pi[0] * Pi[1] / Pi[3]
                    * y1[Vars.t]**(self.__psi+self.__varsigma+4.)
                    * self.__f_tau2over3
                )**(1./(self.__varsigma+1))
        elif self.__transfer == 'thompson':
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
            Values of sigma points for which to solve for `y`. The first point
            should be 0 for `reverse` == False and 1 for `reverse` == True.
            If None than used `__sigma` that is [0,1] for `reverse` == True and
            [1,0] otherwise, this is default value.

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
            # self._dy1,
            self.__Pi0,
            bounds = ((0.1, 10,),) * lenVars
        )
        
        if opt_res.status == 0:
            self.__Pi = opt_res.x
            return opt_res.x
        else:
            raise RuntimeError('Cannon solve optimization problem. Relative OptimizeResult is\n{}'.format(opt_res))

    def mesh(self, n=1000):
        '''
        Distribution of unknown functions of the sigma mesh with `n` points.

        Parameters
        ----------
        n : positive int, optional
            Number of points in sigma mesh

        Returns
        -------
        array, shape(n, len(Pi))
            Array containing the value of `y` for each point of the mesh.            
        '''
        self.getPi()

        if self.__reverse:
            sigma = np.linspace(1, 0, n)
        else:
            sigma = np.linspace(0, 1, n)

        ys = self._integrate(self.__Pi, sigma=sigma)
        return sigma, ys


###################


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        logtau = float(argv[1])
    else:
        logtau = 3
    
    fp = FindPi(
        10**logtau,
        reverse=True,
        heating = 'alpha',
        transfer = 'ff',
#        opacity = (1., 2.5),
    )
    print( fp.getPi() )
    plot_vert_struct(fp)
