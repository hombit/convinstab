#!/usr/bin/env python3

import numpy as np
import logging
from ketsaris_like import FindPi
from scipy import optimize
from multiprocessing import Pool
from sys import argv


logging.basicConfig(level=logging.INFO)

if len(argv) > 1:
    n = int(argv[1])
else:
    n = 100

if len(argv) > 2:
    logging.basicConfig(filename=argv[2])


bounds = np.array( [
    [2, 20], # b
    [0, 10], # d
    [0, 2], # varsigma
    [0, 3.5], # psi
] )

r = np.random.random( size=(4,n) )

x = ( (bounds[:,1] - bounds[:,0]).reshape(-1,1) * r + bounds[:,0].reshape(-1,1) ).T

def dlogT_dlogP_centr(args):
    b, d, varsigma, psi = args
    Pi0 = FindPi._Pi0
    for _ in range(5):
        try:
            fp = FindPi(1e6, Pi0=Pi0, heating=(b,d), transfer='ff', opacity=(varsigma, psi))
            Pi = fp.getPi()
            break
        except RuntimeError as e:
            Pi0 *= 1 + (np.random.rand()-0.5) * 0.1
    else:
        logging.warning(args)
        raise e
    N = Pi[2] * Pi[3] / (Pi[0] * Pi[1]**2)
    return N

with Pool() as pool:
    N = np.array( pool.map(dlogT_dlogP_centr, x), dtype=np.float )

c, dc = optimize.curve_fit(
    lambda x, c_0, *c: c_0 + np.sum( x * c, axis=1 ),
    x,
    N,
    p0 = [1]*5
)
logging.info('c      = {}'.format(c))
logging.info('relerr = {}'.format(np.sqrt(dc.diagonal()) / c ) )

c1 = c.copy()
c1[0] -= 0.4
c1 /= np.min(np.abs(c1))
logging.info('Norm c = {}'.format(c1))
