"""RW mixture CDF/PDF/quantile via C++/GSL numerical integration.

This module is a thin Python wrapper around the shared library compiled from
`RW_inte_cpp.cpp` (built as `RW_inte_cpp.so`). The C++ side uses GSL to:

1) Numerically integrate expressions for the CDF/PDF of the *shifted* RW model,
   using a change-of-variables that maps an improper integral over r∈(0,∞)
   onto a compact interval t∈(0,1).

2) Provide closed-form (incomplete-gamma) expressions for the *standard*
   (non-shifted) Pareto link version, plus a Gaussian-nugget convolution.

The wrapper exposes vectorized NumPy callables:

Shifted (numerical integration on (0,1); no nugget)
    - pRW_transformed_cpp(x, phi, gamma) : CDF
    - dRW_transformed_cpp(x, phi, gamma) : PDF
    - qRW_transformed_cpp(p, phi, gamma) : quantile via Brent

Standard Pareto link (incomplete-gamma; no nugget)
    - pRW_standard_Pareto_vec(x, phi, gamma)
    - dRW_standard_Pareto_vec(x, phi, gamma)
    - qRW_standard_Pareto_vec(p, phi, gamma)

Standard Pareto link with Gaussian nugget (convolution; numerically integrated)
    - pRW_standard_Pareto_nugget_vec(x, phi, gamma, tau)
    - dRW_standard_Pareto_nugget_vec(x, phi, gamma, tau)
    - qRW_standard_Pareto_nugget_vec(p, phi, gamma, tau)

Notes
-----
* This file intentionally keeps the Python layer lightweight: all heavy
  numerical work happens in C++ (GSL integration + root finding), which is
  substantially faster and more stable than pure-Python/scipy alternatives
  for this particular problem.
* The shared library is loaded relative to THIS file, which makes the module
  robust to the current working directory (important for GitHub usage).

Build (macOS/Homebrew example)
------------------------------
    g++ -I/opt/homebrew/include -std=c++11 -Wall -pedantic RW_inte_cpp.cpp \
        -shared -fPIC -L/opt/homebrew/lib -o RW_inte_cpp.so -lgsl -lgslcblas

Linux users typically have includes in /usr/include and libs in /usr/lib.
"""

from __future__ import annotations

import os
import ctypes
import numpy as np
from pathlib import Path


# -----------------------------------------------------------------------------
# Shared library loading
# -----------------------------------------------------------------------------
# We resolve the .so path relative to this file so imports work from anywhere
# (e.g., when this repository is installed as a package or imported from tests).
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_SO = _THIS_DIR / "RW_inte_cpp.so"

try:
    RW_lib = ctypes.CDLL(str(_DEFAULT_SO))
except OSError as e:
    raise OSError(
        f"Failed to load shared library: {_DEFAULT_SO}\n"
        "Make sure you compiled RW_inte_cpp.cpp into RW_inte_cpp.so, and that "
        "GSL is installed and discoverable by the dynamic loader.\n"
        "See the build command in the module docstring."
    ) from e


# -----------------------------------------------------------------------------
# Shifted RW (numerical integration on t in (0,1); no nugget)
# -----------------------------------------------------------------------------
# These correspond to the C++ functions:
#   double pRW_transformed(double x, double phi, double gamma)
#   double dRW_transformed(double x, double phi, double gamma)
#   double qRW_transformed_brent(double p, double phi, double gamma)
#
# In the C++ implementation, the improper integral over r∈(0,∞) is transformed
# using r = (1-t)/t, t∈(0,1), with Jacobian dr = dt / t^2.
RW_lib.pRW_transformed.restype = ctypes.c_double
RW_lib.pRW_transformed.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)

RW_lib.dRW_transformed.restype = ctypes.c_double
RW_lib.dRW_transformed.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)

RW_lib.qRW_transformed_brent.restype = ctypes.c_double
RW_lib.qRW_transformed_brent.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)

# Vectorized wrappers (broadcast like numpy ufuncs; returns float arrays)
pRW_transformed_cpp = np.vectorize(RW_lib.pRW_transformed, otypes=[float])
dRW_transformed_cpp = np.vectorize(RW_lib.dRW_transformed, otypes=[float])
qRW_transformed_cpp = np.vectorize(RW_lib.qRW_transformed_brent, otypes=[float])


# -----------------------------------------------------------------------------
# Standard (non-shifted) Pareto link (incomplete-gamma; no nugget)
# -----------------------------------------------------------------------------
# These correspond to closed-form expressions derived via incomplete gamma
# functions (see the C++ file for details). This avoids numerical integration
# and is typically ~10x faster than integrating the shifted form.
RW_lib.pRW_standard_Pareto_C.restype = ctypes.c_double
RW_lib.pRW_standard_Pareto_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)

RW_lib.dRW_standard_Pareto_C.restype = ctypes.c_double
RW_lib.dRW_standard_Pareto_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)

RW_lib.qRW_standard_Pareto_C_brent.restype = ctypes.c_double
RW_lib.qRW_standard_Pareto_C_brent.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)

dRW_standard_Pareto_vec = np.vectorize(RW_lib.dRW_standard_Pareto_C, otypes=[float])
pRW_standard_Pareto_vec = np.vectorize(RW_lib.pRW_standard_Pareto_C, otypes=[float])
qRW_standard_Pareto_vec = np.vectorize(RW_lib.qRW_standard_Pareto_C_brent, otypes=[float])


# -----------------------------------------------------------------------------
# Standard Pareto link + Gaussian nugget (convolution)
# -----------------------------------------------------------------------------
# Here tau is the nugget SD. The C++ side evaluates the convolution integrals
# numerically (again using GSL), because the nugget breaks the closed form.
RW_lib.dRW_standard_Pareto_nugget_C.restype = ctypes.c_double
RW_lib.dRW_standard_Pareto_nugget_C.argtypes = (
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
)

RW_lib.pRW_standard_Pareto_nugget_C.restype = ctypes.c_double
RW_lib.pRW_standard_Pareto_nugget_C.argtypes = (
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
)

RW_lib.qRW_standard_Pareto_nugget_C_brent.restype = ctypes.c_double
RW_lib.qRW_standard_Pareto_nugget_C_brent.argtypes = (
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
)

dRW_standard_Pareto_nugget_vec = np.vectorize(RW_lib.dRW_standard_Pareto_nugget_C, otypes=[float])
pRW_standard_Pareto_nugget_vec = np.vectorize(RW_lib.pRW_standard_Pareto_nugget_C, otypes=[float])
qRW_standard_Pareto_nugget_vec = np.vectorize(RW_lib.qRW_standard_Pareto_nugget_C_brent, otypes=[float])


# -----------------------------------------------------------------------------
# Historical / alternative implementations (kept for reference)
# -----------------------------------------------------------------------------
# The remainder of the original file contained mpmath/scipy implementations
# of the same objects. They are commented out because:
#   (i) they were slower by a wide margin, and
#   (ii) root finding for the quantile could be numerically unreliable in scipy
#        for extreme p (near 0 or 1) and for certain (phi, gamma) combinations.
#
# If you need a pure-Python fallback in the future (e.g., Windows without a
# compiler toolchain), you can resurrect those blocks and provide a runtime
# switch, but for research-grade use the C++/GSL backend is recommended.


# -----------------------------------------------------------------------------
# Reference implementations (commented out)
# -----------------------------------------------------------------------------
# dRW_cpp = model_sim.dRW
# pRW_cpp = model_sim.pRW
# def qRW_cpp(p, phi, gamma):
#     return(model_sim.qRW_Newton(p, phi, gamma, 100)) # qRW_Newton is vectorized


# %%
###########################################################
# Integration with mpmath
# for the shifted Pareto, no nugget
# pRW_mpmath, dRW_mpmath, qRW_mpmath
###########################################################
# mp.dps = 15

# # mpmath dRW
# def dRW_integrand_mpmath(r, x, phi, gamma):
#     numerator = mp.power(r, phi-1.5)
#     denominator = mp.power(x + mp.power(r, phi), 2)
#     exp = mp.exp(-(gamma/(2*r)))
#     return numerator / denominator * exp
# def dRW_mpmath(x, phi, gamma, **kwargs):
#     return mp.sqrt(gamma/(2 * mp.pi)) * mp.quad(lambda r : dRW_integrand_mpmath(r, x, phi, gamma), [0, mp.inf], method='tanh-sinh', **kwargs)
# dRW_mpmath_vec = np.vectorize(dRW_mpmath) # return a np.array of mpf
# def dRW_mpmath_vec_float(x, phi, gamma): # return a np.array of floats
#     return(dRW_mpmath_vec(x, phi, gamma).astype(float))

# # mpmath pRW
# def pRW_integrand_mpmath(r, x, phi, gamma):
#     numerator = mp.power(r, phi-1.5)
#     denominator = x + mp.power(r, phi)
#     exp = mp.exp(-(gamma/(2*r)))
#     return numerator / denominator * exp
# def pRW_mpmath(x, phi, gamma):
#     return 1 - mp.sqrt(gamma/(2*mp.pi)) * mp.quad(lambda r : pRW_integrand_mpmath(r, x, phi, gamma), [0, mp.inf], method='tanh-sinh')
# pRW_mpmath_vec = np.vectorize(pRW_mpmath) # return a np.array of mpf
# def pRW_mpmath_vec_float(x, phi, gamma): # return a np.array of floats
#     return(pRW_mpmath_vec(x, phi, gamma).astype(float))

# # mpmath transform dRW -- no significant gain in terms of accuracy as compared to dRW_mpmath
# # mpmath with high dps can handle integration from [0, mp.inf] well
# def dRW_integrand_transformed_mpmath(t, x, phi, gamma):
#     numerator = mp.power((1-t)/t, phi-1.5)
#     denominator = mp.power(x + mp.power((1-t)/t, phi), 2)
#     exp = mp.exp(-gamma/(2*(1-t)/t))
#     jacobian = 1 / mp.power(t, 2)
#     return (numerator / denominator) * exp * jacobian
# def dRW_transformed_mpmath(x, phi, gamma):
#     return mp.sqrt(gamma/(2 * mp.pi)) * mp.quad(lambda t : dRW_integrand_transformed_mpmath(t, x, phi, gamma), [0, 1], method='tanh-sinh')

# # mpmath transform pRW -- no significant gain in terms of accuracy, as compared to pRW_mpmath
# # mpmath with high dps can handle integration from [0, mp.inf] well
# def pRW_integrand_transformed_mpmath(t, x, phi, gamma):
#     numerator = mp.power((1-t)/t, phi-1.5)
#     denominator = x + mp.power((1-t)/t, phi)
#     exp = mp.exp(- gamma / (2 * (1-t)/t))
#     jacobian = 1 / mp.power(t, 2)
#     return numerator / denominator * exp * jacobian
# def pRW_transformed_mpmath(x, phi, gamma):
#     return 1 - mp.sqrt(gamma/(2*mp.pi)) * mp.quad(lambda t: pRW_integrand_transformed_mpmath(t, x, phi, gamma), [0, 1], method='tanh-sinh')

# # mpmath quantile function
# def qRW_mpmath(p, phi, gamma):
#     return mp.findroot(lambda x : pRW_mpmath(x, phi, gamma) - p,
#                        [0,1e12],
#                        solver='anderson')
# qRW_mpmath_vec = np.vectorize(qRW_mpmath) # return a np.array of mpf
# def qRW_mpmath_vec_float(p, phi, gamma): # return a np.array of floats
#     return(qRW_mpmath_vec(p, phi, gamma).astype(float))

# %%
###########################################################
# Integration with scipy QUAD                             #
# for the shifted Pareto, no nugget                       #
# pRW_scipy, dRW_scipy, qRW_scipy                         #
###########################################################

# # scipy dRW
# @jit(nopython=True)
# def dRW_integrand_scipy(r, x, phi, gamma):
#     return (r**(phi-3/2)) / ((x+r**phi)**2) * np.exp(-(gamma/(2*r)))
# def dRW_scipy(x, phi, gamma):
#     return np.sqrt(gamma/(2*np.pi)) * quad(dRW_integrand_scipy, 0, np.inf, args=(x, phi, gamma))[0]
# dRW_scipy_vec = np.vectorize(dRW_scipy, otypes=[float])

# # scipy pRW
# @jit(nopython=True)
# def pRW_integrand_scipy(r, x, phi, gamma):
#     numerator = np.power(r, phi-1.5)
#     denominator = x + np.power(r, phi)
#     exp = np.exp(-(gamma/(2*r)))
#     return numerator / denominator * exp
# def pRW_scipy(x, phi, gamma):
#     return 1 - np.sqrt(gamma/(2*np.pi)) * quad(pRW_integrand_scipy, 0, np.inf, args=(x, phi, gamma))[0]
# pRW_scipy_vec = np.vectorize(pRW_scipy, otypes=[float]) # return a np.array of mpf

# # scipy dRW transformed between [0,1]
# @jit(nopython=True)
# def dRW_integrand_transformed_scipy(t, x, phi, gamma):
#     ratio_numerator = np.power((1-t)/t, phi-1.5)
#     ratio_denominator = (x + np.power((1-t)/t, phi))**2
#     exponential_term = np.exp(-gamma/(2*((1-t)/t)))
#     jacobian = 1/(t**2)
#     return (ratio_numerator/ratio_denominator) * exponential_term * jacobian
# def dRW_transformed_scipy(x, phi, gamma):
#     return np.sqrt(gamma/(2*np.pi)) * quad(dRW_integrand_transformed_scipy, 0, 1, args=(x, phi, gamma))[0]
# dRW_transformed_scipy_vec = np.vectorize(dRW_transformed_scipy, otypes=[float])

# # scipy pRW transformed between [0,1]
# @jit(nopython=True)
# def pRW_integrand_transformed_scipy(t, x, phi, gamma):
#     numerator = np.power((1-t)/t, phi-1.5)
#     denominator = x + np.power((1-t)/t, phi)
#     exp = np.exp(- gamma / (2 * (1-t)/t))
#     jacobian = 1 / np.power(t, 2)
#     return numerator / denominator * exp * jacobian
# def pRW_transformed_scipy(x, phi, gamma):
#     return 1 - np.sqrt(gamma/(2*np.pi)) * quad(pRW_integrand_transformed_scipy, 0, 1, args=(x, phi, gamma))[0]
# pRW_transformed_scipy_vec = np.vectorize(pRW_transformed_scipy, otypes=[float])

# def qRW_scipy(p, phi, gamma):
#     try:
#         return scipy.optimize.root_scalar(lambda x: pRW_transformed_scipy(x, phi, gamma) - p,
#                                         bracket=[0, 1e12],
#                                         fprime = lambda x: dRW_transformed_scipy(x, phi, gamma),
#                                         x0 = 10,
#                                         method = 'ridder').root
#     except Exception as e:
#         print(e)
#         print('p=',p,',','phi=',phi,',','gamma',gamma)
# qRW_scipy_vec = np.vectorize(qRW_scipy, otypes=[float])

# %%
###########################################################
# Incomplete gamma functions with GSL                     #
# for the nonshifted Pareto, no nugget                    #
# pRW_stdPareto, dRW_stdPareto, qRW_stdPareto             #
###########################################################

# # using the GSL incomplete gamma function is much (10x) faster than mpmath
# upper_gamma_C = np.vectorize(model_sim.lib.upper_gamma_C, otypes = [float])
# lower_gamma_C = np.vectorize(model_sim.lib.lower_gamma_C, otypes = [float])
# def dRW_stdPareto(x, phi, gamma):
#     upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2*np.power(x, 1/phi)))
#     # upper_gamma = float(mpmath.gammainc(0.5 - phi, a = gamma / (2*np.power(x, 1/phi))))
#     return (1/np.power(x,2)) * np.sqrt(1/np.pi) * np.power(gamma/2, phi) * upper_gamma
# def pRW_stdPareto(x, phi, gamma):
#     lower_gamma = lower_gamma_C(0.5, gamma / (2*np.power(x, 1/phi)))
#     upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2*np.power(x, 1/phi)))
#     # lower_gamma = float(mpmath.gammainc(0.5, b = gamma / (2*np.power(x, 1/phi))))
#     # upper_gamma = float(mpmath.gammainc(0.5 - phi, a = gamma / (2*np.power(x, 1/phi))))
#     survival = np.sqrt(1/np.pi) * lower_gamma + (1/x) * np.sqrt(1/np.pi) * np.power(gamma/2, phi) * upper_gamma
#     return 1 - survival
# def qRW_stdPareto(p, phi, gamma):
#     try:
#         return scipy.optimize.root_scalar(lambda x: pRW_stdPareto(x, phi, gamma) - p,
#                                         bracket=[0.1,1e12],
#                                         fprime = lambda x: dRW_stdPareto(x, phi, gamma),
#                                         x0 = 10,
#                                         method='newton').root
#     except Exception as e:
#         print(e)
#         print('p=',p,',','phi=',phi,',','gamma',gamma)
# dRW_stdPareto_vec = np.vectorize(dRW_stdPareto, otypes=[float])
# pRW_stdPareto_vec = np.vectorize(pRW_stdPareto, otypes=[float])
# qRW_stdPareto_vec = np.vectorize(qRW_stdPareto, otypes=[float])
