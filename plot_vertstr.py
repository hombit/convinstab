#!/usr/bin/env python3


import matplotlib.pyplot as plt
from matplotlib import rc
from vertstr import *
from fractions import Fraction


rc('text', usetex=True)

dashes = [
    (10000,1),
    (5,2,10,5),
    (2,6),
    (4,8,4,8),
    (2,2,),
    (2,4,2,4,2,8)
]



def _title(fp):
    return r'''$b = {b}$, $d = {d}$, $\varsigma = {varsigma}$, $\psi = {psi}$, $\log {tau_name} = {logtau:.1g}$
{Pi}'''.format(
            b = Fraction(fp.b),
            d = Fraction(fp.d),
            varsigma = Fraction(fp.varsigma),
            psi = Fraction(fp.psi),
            tau_name = r'\tau_0' if fp.transfer == 'absorption' else r'\delta',
            logtau = np.log10(fp.tau),
            xi = np.sqrt(fp.Pi[0] * 5./3.),
            Pi = ',  '.join( map(
                lambda i: r'$\Pi_{} = {:.3f}$'.format(i+1, fp.Pi[i]),
                range(fp.Pi.shape[0])
            ) )
        )


def plot_to_ax(ax, fp, n=10000, entropy=False, dlogTdlogP=False):
    sigma, ys = fp.pzqt(np.linspace(1,0,n)**0.3)

    ax.set_title(_title(fp), multialignment = 'center')
    ax.set_xlabel(r'$x$')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    lines = []
    lines += ax.plot(ys[:,Vars.z], ys[:,Vars.p], label=r'$p$')
    lines += ax.plot(ys[:,Vars.z], sigma,        label=r'$\sigma$')
    lines += ax.plot(ys[:,Vars.z], ys[:,Vars.q], label=r'$q$')
    lines += ax.plot(ys[:,Vars.z], ys[:,Vars.t], label=r'$t$')
    if entropy:
        lines += ax.plot(
            ys[:,Vars.z],
            entropy_tp(ys[:,Vars.t], ys[:,Vars.p]),
            label=r'$s$'
        )
    if dlogTdlogP:
        lines += ax.plot(
            ys[:,Vars.z],
            dlogT_dlogP(ys[:,Vars.t], ys[:,Vars.p]),
            label=r'$d \log{T} / d \log{P}$'
        )
    plt.setp(lines, color='k')
    for i, line in enumerate(lines):
        plt.setp(line, dashes=dashes[i])
    return lines


def plot_to_file(fp, filename=None, **kwargs):
    '''
    Plot distributions of various functions to the EPS file.
    This function plots all unknown functions, and optionally normalized
    entropy `s` and derivative d log(T) / d log(P).

    Parameters
    ----------
    fp : FindPi
        Object used to calculate variables.
    n : positive int, optional
        Number of sigma points.
    filename : str or None, optional
        Path of the filename to save plot. If None construct filename of
        the format ``{heating}_{transfer}_logtau_{logtau}.eps`` in the
        local directory.
    entropy : bool, optional
        Plot entropy normalized on maximum absolute value.
    dlogTdlogP : bool, optional
        Plot d log(T) / d log(P). When its value larger than 0.4 convection
        appears.
    '''
    if filename is None:
        filename = '{}_{}_logtau_{}.eps'.format(fp.heating, fp.transfer, np.log10(fp.tau))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_to_ax(ax, fp, **kwargs)
    plt.legend(loc='best')
    plt.savefig(filename)


def plot_four_to_file(fps, filename=None, **kwargs):
    rc('font', size=8)
    fig = plt.figure()
    for i, fp in enumerate(fps):
        ax = fig.add_subplot(2, 2, i+1)
        lines = plot_to_ax(ax, fp, **kwargs)
    fig.legend(
        lines,
        (plt.getp(line, 'label') for line in lines),
        loc = 'lower center',
        ncol = 6,
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.,
        frameon = True,
    )
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(bottom=0.105)
    plt.savefig(filename)


def plot_four_for_paper(heating, **kwargs):
    filename = '{}.eps'.format(heating)
    fps = []
    for tau in (1e1, 1e6):
        for transfer in ('absorption', 'scattering'):
            fps.append( FindPi(
                tau,
                Pi0=[2, 0.7, 1, 0.6],
                heating=heating,
                transfer=transfer
            ) )
    plot_four_to_file(fps, filename, dlogTdlogP=True, entropy=False)



##########################



if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        logtau = float(argv[1])
    else:
        logtau = 3
    
    fp = FindPi(
        10**logtau,
#        Pi0=[6., 0.6, 1., 0.5],
        reverse=True,
        heating = (15., 0.),
        transfer = 'scattering',
#        opacity = (1., 2.5),
    )
    print( fp.getPi() )
    plot_to_file(fp, dlogTdlogP=True, entropy=False)
    #for heating in ('alpha', 'ion'):
    #    plot_four_for_paper(heating)
