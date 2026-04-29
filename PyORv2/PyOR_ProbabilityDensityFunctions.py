"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module provides probability density functions (PDFs) for use in 
    magnetic resonance simulations and statistical modeling within PyOR.

    Functions include Gaussian distributions, Lorentzian distributions, 
    and custom probability models relevant for signal analysis and noise modeling.
"""


import numpy as np

def PDFgaussian(x: np.ndarray, std: float, mean: float) -> np.ndarray:
    """
    Compute the normalized Gaussian (normal) probability density function.

    Parameters
    ----------
    x : np.ndarray
        Array of input values (the variable over which the Gaussian is evaluated).
    std : float
        Standard deviation of the Gaussian distribution.
    mean : float
        Mean (center) of the Gaussian distribution.

    Returns
    -------
    np.ndarray
        The normalized Gaussian probability density values corresponding to `x`.

    Notes
    -----
    The Gaussian function is defined as:
        PDF(x) = (1 / sqrt(2πσ²)) * exp(- (x - μ)² / (2σ²))

    The output is normalized such that the sum of the returned values equals 1.
    This is useful for discrete approximations of continuous distributions.
    """
    gaussian = (1 / np.sqrt(2 * np.pi * std ** 2)) * np.exp(-1 * (x - mean) ** 2 / (2 * std ** 2))
    return gaussian / np.sum(gaussian)
