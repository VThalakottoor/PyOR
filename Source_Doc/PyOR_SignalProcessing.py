"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module provides functions for signal processing in magnetic resonance 
    simulations, including time-domain and frequency-domain transformations.

    Functions include Fourier transforms, filtering operations, phase corrections, 
    and signal normalization techniques tailored for NMR and EPR data.
"""


import numpy as np

def WindowFunction(t, signal, LB):
    """
    Applies an exponential window function to simulate signal decay.

    This function multiplies a time-domain signal by an exponential decay factor,
    typically used in signal processing to apply line broadening in NMR and other spectroscopy techniques.

    Parameters
    ----------
    t : array-like
        Time array (same length as the signal).
    signal : array-like
        The input signal array to be decayed.
    LB : float
        Line broadening factor (decay rate).

    Returns
    -------
    array-like
        The decayed signal after applying the exponential window.
    """
    window = np.exp(-LB * t)
    return signal * window


def FourierTransform(signal, fs, zeropoints):
    """
    Computes the Fourier Transform of a time-domain signal with optional zero filling.

    This function performs a one-dimensional FFT of a signal and returns the 
    frequency axis and the corresponding complex spectrum. The result is centered 
    using `fftshift`, and zero-filling is applied to improve spectral resolution.

    Parameters
    ----------
    signal : array-like
        Time-domain signal (1D array).
    fs : float
        Sampling frequency in Hz (not angular). This represents the total bandwidth.
    zeropoints : int
        Zero-filling factor. Total FFT points = zeropoints × len(signal).

    Returns
    -------
    freq : ndarray
        Frequency axis in Hz, centered around 0 using fftshift.
    spectrum : ndarray
        Complex frequency-domain representation of the input signal.
    """
    signal[0] = signal[0]  # Placeholder to preserve first point (no effect)
    spectrum = np.fft.fft(signal, zeropoints * signal.shape[-1])
    spectrum = np.fft.fftshift(spectrum)
    freq = np.linspace(-fs / 2, fs / 2, spectrum.shape[-1])
    return freq, spectrum

    
def PhaseAdjust_PH0(spectrum, PH0):
    """
    Applies zero-order phase correction to a spectrum.

    This function performs a uniform phase shift (PH0) across the entire 
    frequency-domain spectrum. This is typically used to correct for a 
    constant phase error introduced by the acquisition system or processing.

    Parameters
    ----------
    spectrum : ndarray
        Complex spectrum (1D or 2D array).
    PH0 : float
        Zero-order phase in degrees.

    Returns
    -------
    ndarray
        Phase-adjusted spectrum.
    """
    phase_rad = np.deg2rad(PH0)  # Convert degrees to radians
    return spectrum * np.exp(1j * phase_rad)


def PhaseAdjust_PH1(freq, spectrum, pivot, slope):
    """
    Applies first-order phase correction to a spectrum.

    First-order phase correction applies a frequency-dependent phase shift, 
    typically to correct dispersion lineshape distortions. The phase is 
    zero at the `pivot` frequency and changes linearly with the slope.

    Parameters
    ----------
    freq : ndarray
        Frequency axis (same length as the spectrum).
    spectrum : ndarray
        Complex-valued frequency-domain spectrum.
    pivot : float
        Frequency (in Hz or ppm) at which the phase correction is zero.
    slope : float
        Slope of the phase correction in degrees per kHz.

    Returns
    -------
    ndarray
        Spectrum after applying first-order phase correction.
    """
    freq_axis = np.arange(len(freq))
    pivot_index = np.searchsorted(freq, pivot)
    PH1 = -1.0e-3 * slope * (freq_axis - pivot_index)  # in degrees
    phase_rad = np.deg2rad(PH1)  # convert to radians
    return spectrum * np.exp(1j * phase_rad)


def FourierTransform2D(signal, fs1, fs2, zeropoints):
    """
    Perform a 2D Fourier Transform on a 2D signal array.

    This function is commonly used in 2D NMR to transform a time-domain 
    signal into the frequency domain along both the indirect (F1) and 
    direct (F2) dimensions.

    Parameters
    ----------
    signal : ndarray (2D)
        2D signal array with dimensions [F2, F1] (rows: direct, cols: indirect).
    fs1 : float
        Sampling frequency for the F1 (indirect) dimension.
    fs2 : float
        Sampling frequency for the F2 (direct) dimension.
    zeropoints : int
        Zero-filling factor, multiplies each dimension size for padding.

    Returns
    -------
    freq1 : ndarray
        Frequency axis for the F1 (indirect) dimension.
    freq2 : ndarray
        Frequency axis for the F2 (direct) dimension.
    spectrum : ndarray (2D)
        Complex-valued frequency-domain spectrum after 2D FFT and shift.
    """
    signal[:, 0] = signal[:, 0] / 2  # apply half value correction at t=0 (F2 dim)
    
    shape_f1 = zeropoints * signal.shape[1]
    shape_f2 = zeropoints * signal.shape[0]
    
    spectrum = np.fft.fft2(signal, s=(shape_f2, shape_f1), axes=(0, 1))
    spectrum = np.fft.fftshift(spectrum)

    freq1 = np.linspace(-fs1 / 2, fs1 / 2, spectrum.shape[1])
    freq2 = np.linspace(-fs2 / 2, fs2 / 2, spectrum.shape[0])

    return freq1, freq2, spectrum


def FourierTransform2D_F1(signal, fs, zeropoints):
    """
    Perform a 1D Fourier Transform along the F1 (indirect) dimension 
    of a 2D NMR signal.

    This function applies a 1D FFT to each column (F1 dimension) of 
    the signal matrix and returns the frequency axis and spectrum.

    Parameters
    ----------
    signal : ndarray (2D)
        2D time-domain signal array with dimensions [F2, F1].
    fs : float
        Sampling frequency for the F1 dimension.
    zeropoints : int
        Zero-filling factor for F1 dimension (not used here).

    Returns
    -------
    freq : ndarray
        Frequency axis for the F1 (indirect) dimension.
    spectrum : ndarray (2D)
        Transformed signal with FFT applied along F1.
    """
    spectrum = np.zeros_like(signal, dtype=np.cdouble)
    for i in range(signal.shape[-1]):
        spec = np.fft.fft(signal[:, i])
        spectrum[:, i] = np.fft.fftshift(spec)

    freq = np.linspace(-fs / 2, fs / 2, spectrum.shape[0])
    return freq, spectrum


def FourierTransform2D_F2(signal, fs, zeropoints):
    """
    Perform a 1D Fourier Transform along the F2 (direct) dimension 
    of a 2D NMR signal.

    Applies FFT row-wise (along the F2 axis) and returns the frequency axis 
    and the resulting spectrum.

    Parameters
    ----------
    signal : ndarray
        2D time-domain signal array with dimensions [F2, F1].
    fs : float
        Sampling frequency for the F2 (direct) dimension.
    zeropoints : int
        Zero-filling factor to improve frequency resolution (multiplied to original length).

    Returns
    -------
    freq : ndarray
        Frequency axis corresponding to the F2 dimension.
    spectrum : ndarray
        Fourier-transformed 2D spectrum along F2.
    """
    # Optional: Scaling the first point (if needed in processing)
    signal[0] = signal[0] / 2

    # Apply FFT along axis 1 (F2 direction), with zero-filling
    spectrum = np.fft.fft(signal, zeropoints * signal.shape[-1], axis=1)
    spectrum = np.fft.fftshift(spectrum, axes=1)

    # Generate frequency axis
    freq = np.linspace(-fs / 2, fs / 2, spectrum.shape[-1])

    return freq, spectrum
