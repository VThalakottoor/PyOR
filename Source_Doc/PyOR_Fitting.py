"""
PyOR - Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
Email: vineethfrancis.physics@gmail.com

This file contains functions for curve fitting.

Documentation is done.
"""

import numpy as np
from scipy.optimize import curve_fit

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Curve Fitting Functions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def Exp_Decay(x, a, b, c):
    """
    Single exponential decay function.

    Model
    -----
    f(x) = a * exp(-b * x) + c

    Parameters
    ----------
    x : array_like
        Independent variable.
    a, b, c : float
        Fit parameters.

    Returns
    -------
    array_like
        Evaluated function.
    """
    return a * np.exp(-b * x) + c


def Exp_Decay_2(x, a, b, c, d, e):
    """
    Double exponential decay function.

    Model
    -----
    f(x) = a * exp(-b * x) + c * exp(-d * x) + e

    Parameters
    ----------
    x : array_like
        Independent variable.
    a, b, c, d, e : float
        Fit parameters.

    Returns
    -------
    array_like
        Evaluated function.
    """
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e


def Exp_Buildup(x, a, b, c):
    """
    Exponential build-up function.

    Model
    -----
    f(x) = c - (c - a) * exp(-b * x)

    Parameters
    ----------
    x : array_like
        Independent variable.
    a, b, c : float
        Fit parameters.

    Returns
    -------
    array_like
        Evaluated function.
    """
    return c - (c - a) * np.exp(-b * x)


def Fitting_LeastSquare(func, xdata, ydata):
    """
    Perform non-linear least squares fitting using a custom function.

    Parameters
    ----------
    func : callable
        Model function to fit, with signature `func(x, ...)`.
    xdata : array_like
        Independent data (x-values).
    ydata : array_like
        Dependent data (y-values).

    Returns
    -------
    popt : array
        Optimal values for the parameters.
    pcov : 2D array
        Estimated covariance of popt.

    Notes
    -----
    For more details, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    """
    popt, pcov = curve_fit(func, xdata, ydata)
    return popt, pcov
