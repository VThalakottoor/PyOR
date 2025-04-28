"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module provides the `MaserDataAnalyzer` class for loading, analyzing,
    and visualizing maser signal data in both time and frequency domains.

    It includes interactive matplotlib visualizations for signal inspection,
    FFT/iFFT transformations, and automatic subplot saving after user interactions.
"""


import numpy as np
import matplotlib.pyplot as plt
import os

def Bruker1Ddata(filepath, outname):
    """
    Converts a Bruker 'fid' file into a CSV with Mx and My columns only.

    Args:
        filepath (str): Path to the Bruker 'fid' binary file.
        outname (str): Output CSV filename (with or without .csv extension).
    """
    # Ensure correct output file extension
    if not outname.lower().endswith(".csv"):
        outname += ".csv"

    # Step 1: Load int32 binary data from file
    dat = np.fromfile(filepath, dtype=np.int32)

    # Step 2: Separate real (Mx) and imaginary (My) parts
    mx = dat[0::2]
    my = dat[1::2]

    # Step 3: Stack Mx and My into two-column array
    out_array = np.column_stack((mx, my))

    # Step 4: Save to CSV (no header, comma delimiter)
    np.savetxt(outname, out_array, delimiter=",", fmt="%.6f")

import numpy as np

def Bruker2Ddata(filepath, TD_F1, FIDno, outname):
    """
    Extract and save a specific FID (Free Induction Decay) from Bruker 2D NMR data.

    This function reads a Bruker 2D "ser" file, extracts the specified FID number,
    and saves its real and imaginary components into a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the Bruker "ser" file containing the raw 2D NMR data.
    TD_F1 : int
        The size of the indirect dimension (F1) of the dataset.
    FIDno : int
        The number of the FID to extract (0-based indexing).
    outname : str
        The desired output CSV file name. 
        If the extension is not provided, ".csv" will be appended automatically.

    Notes
    -----
    The output CSV will contain two columns:
    - First column: Real part of the FID.
    - Second column: Imaginary part of the FID.
    Each row corresponds to a point in the FID.
    """
    # Ensure the output filename has the .csv extension
    if not outname.lower().endswith(".csv"):
        outname += ".csv"

    # Load the full dataset as 32-bit integers
    dat = np.fromfile(filepath, dtype=np.int32)

    # Separate real and imaginary parts
    Mx = dat[0::2].reshape((TD_F1, -1))  # Real parts
    My = dat[1::2].reshape((TD_F1, -1))  # Imaginary parts

    # Extract the specific FID number
    real_part = Mx[FIDno,:]
    imag_part = My[FIDno,:]

    # Stack real and imaginary parts side-by-side
    out_array = np.column_stack((real_part, imag_part))

    # Save to CSV
    np.savetxt(outname, out_array, delimiter=",", fmt="%.6f")

class MaserDataAnalyzer:
    """
    MaserDataAnalyzer handles loading, processing, and plotting of maser data.

    Attributes:
        filepath (str): Path to the CSV file containing maser signal data.
        offset (float): Frequency offset for spectrum display.
        flip_spectrum (bool): Flag to reverse the spectrum.
        abs_spectrum (bool): Flag to display absolute value of the FFT.
        dt (float): Time step between signal points.
    """

    def __init__(self, filepath, dt, offset=0.0, flip_spectrum=False, abs_spectrum=True, simulation = False):
        self.filepath = filepath
        self.offset = offset
        self.dt = dt
        self.flip_spectrum = flip_spectrum
        self.abs_spectrum = abs_spectrum
        self.simulation = simulation
        self.Xlimt = None

        self.Load_Data()
        self.Prepare_Signal()
        self.Compute_FFT()

    def Plot(self):
        self.Setup_Plot()
        self.Connect_Events()

    def Load_Data(self):
        """Loads maser signal data from a CSV file."""
        if self.simulation:
            self.data = np.load(self.filepath)
            self.Mx = self.data.real
            self.My = self.data.imag           
        else:
            self.data = np.genfromtxt(self.filepath, delimiter=',')
            self.Mx = self.data[:, 0]
            self.My = self.data[:, 1]
        self.tpoints =  np.linspace(0, self.Mx.shape[-1] * self.dt, self.Mx.shape[-1] )  
        self.fs = 1.0 / self.dt

    def Prepare_Signal(self):
        """Creates a complex signal from Mx and My components."""
        self.signal = self.Mx + 1j * self.My

    def Compute_FFT(self):
        """Computes and prepares the FFT spectrum for display."""
        Spectrum = np.fft.fft(self.signal)
        Spectrum = np.fft.fftshift(Spectrum)
        self.spectrum = Spectrum[::-1] if self.flip_spectrum else Spectrum
        self.freq = np.linspace(-self.fs / 2, self.fs / 2, self.signal.shape[-1]) + self.offset

    def Setup_Plot(self):
        """Creates and configures a 2x2 subplot grid."""
        self.figsize = (12, 9)
        self.fig, self.ax = plt.subplots(2, 2, figsize=self.figsize)

        # Time domain
        self.line1, = self.ax[0, 0].plot(self.tpoints, self.signal.real, '-', color='green')
        self.ax[0, 0].set_title("Time Domain")
        self.ax[0, 0].set_xlabel("Time [s]")
        self.ax[0, 0].set_ylabel("Signal")
        self.ax[0, 0].grid()

        # Frequency domain (top)
        self.vline1 = self.ax[0, 1].axvline(color='k', lw=0.8, ls='--')
        self.vline2 = self.ax[0, 1].axvline(color='k', lw=0.8, ls='--')
        self.text1 = self.ax[0, 1].text(0.0, 0.95, '', transform=self.ax[0, 1].transAxes)
        spectrum_data = np.abs(self.spectrum) if self.abs_spectrum else self.spectrum
        self.line2, = self.ax[0, 1].plot(self.freq, spectrum_data, '-', color='green')
        self.ax[0, 1].set_title("Frequency Domain (Top)")
        self.ax[0, 1].set_xlabel("Frequency [Hz]")
        self.ax[0, 1].set_ylabel("Spectrum")
        if self.Xlimt is not None:
            self.ax[0, 1].set_xlim(self.Xlimt)
        self.ax[0, 1].grid()

        # Frequency domain (bottom)
        self.line3, = self.ax[1, 0].plot(self.freq, spectrum_data, '-', color='green')
        self.ax[1, 0].set_title("Frequency Domain (Bottom)")
        self.ax[1, 0].set_xlabel("Frequency [Hz]")
        self.ax[1, 0].set_ylabel("Spectrum")
        if self.Xlimt is not None:
            self.ax[1, 0].set_xlim(self.Xlimt)
        self.ax[1, 0].grid()

        # Reconstructed signal
        self.vline3 = self.ax[1, 1].axvline(color='k', lw=0.8, ls='--')
        self.vline4 = self.ax[1, 1].axvline(color='k', lw=0.8, ls='--')
        self.text2 = self.ax[1, 1].text(0.0, 0.95, '', transform=self.ax[1, 1].transAxes)
        self.line4, = self.ax[1, 1].plot(self.tpoints, self.signal.real, '-', color='green')
        self.ax[1, 1].set_title("Reconstructed Signal")
        self.ax[1, 1].set_xlabel("Time [s]")
        self.ax[1, 1].set_ylabel("Signal")
        self.ax[1, 1].grid()

    def Connect_Events(self):
        """Binds mouse events for interaction."""
        self.fourier = Fourier(self.Mx, self.My, self.spectrum, self.ax, self.fig,
                               self.line1, self.line2, self.line3, self.line4,
                               self.vline1, self.vline2, self.vline3, self.vline4,
                               self.text1, self.text2, self.offset, self.flip_spectrum, self.abs_spectrum, self.filepath)

        self.fig.canvas.mpl_connect("button_press_event", self.fourier.button_press)
        self.fig.canvas.mpl_connect("button_release_event", self.fourier.button_release)

    def Show(self):
        """Displays the interactive plot window."""
        plt.tight_layout()
        plt.show()

    def Plot_Signal(self):
        """Plots the time-domain signal only."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.tpoints, self.signal.real, label="Real Part")
        plt.plot(self.tpoints, self.signal.imag, label="Imaginary Part", linestyle='--')
        plt.title("Time-Domain Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Save the figure to the same directory
        directory = os.path.dirname(self.filepath)
        output_path = os.path.join(directory, "Signal.svg")
        plt.savefig(output_path, format='svg')

    def Plot_FFT(self):
        """Plots the frequency-domain spectrum only."""
        spectrum_data = np.abs(self.spectrum) if self.abs_spectrum else self.spectrum
        plt.figure(figsize=(10, 4))
        plt.plot(self.freq, spectrum_data, color='purple')
        plt.title("Frequency-Domain Spectrum")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude" if self.abs_spectrum else "Complex Value")
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Save the figure to the same directory
        directory = os.path.dirname(self.filepath)
        output_path = os.path.join(directory, "Spectrum.svg")
        plt.savefig(output_path, format='svg')

    def Plot_Mz(self, Mz_list):
        """
        Plot multiple Mz arrays and save the combined figure.

        Parameters:
            Mz_list (list of str):
                A list of base names of `.npy` files (without the `.npy` extension),
                located in the same directory as `self.filepath`.
                Each `.npy` file contains a 1D array representing Mz.

        Returns:
            None
        """
        # Determine the directory containing the .npy files
        directory = os.path.dirname(self.filepath)

        # Create a new figure for plotting
        plt.figure(figsize=(10, 6))

        # Iterate over each Mz file name in the list
        for name in Mz_list:
            # Construct the full path to the .npy file
            file_path = os.path.join(directory, f"{name}.npy")

            # Check if the file exists
            if os.path.exists(file_path):
                # Load the data from the .npy file
                data = np.load(file_path)

                # Plot the data with a label
                plt.plot(self.tpoints, data, label=name)
            else:
                print(f"Warning: File {file_path} does not exist.")

        # Add labels and title to the plot
        plt.xlabel("Time (s)")
        plt.ylabel("Mz")
        plt.title("Mz Plots")
        plt.legend()
        plt.grid(True)

        # Save the figure to the same directory
        output_path = os.path.join(directory, "Mz_plot.svg")
        plt.savefig(output_path, format='svg')

    def Plot_Mx(self, Mx_list):
        """
        Plot all the Mx arrays from a list of file names and save the figure.

        Parameters:
            Mx_list (list of str):
                List of base names of `.npy` files (without the `.npy` extension),
                located in the same directory as `self.filepath`.

        Returns:
            None
        """
        # Determine the directory containing the .npy files
        directory = os.path.dirname(self.filepath)

        # Create a new figure for plotting
        plt.figure(figsize=(10, 6))

        # Iterate over each Mz file name in the list
        for name in Mx_list:
            # Construct the full path to the .npy file
            file_path = os.path.join(directory, f"{name}.npy")

            # Check if the file exists
            if os.path.exists(file_path):
                # Load the data from the .npy file
                data = np.load(file_path)

                # Plot the data with a label
                plt.plot(self.tpoints, data, label=name)
            else:
                print(f"Warning: File {file_path} does not exist.")

        # Add labels and title to the plot
        plt.xlabel("Time (s)")
        plt.ylabel("Mx")
        plt.title("Mx Plots")
        plt.legend()
        plt.grid(True)

        # Save the figure to the same directory
        output_path = os.path.join(directory, "Mx_plot.svg")
        plt.savefig(output_path, format='svg')

class Fourier:
    """
    Fourier handles interactive user selections and signal processing
    for visualizing and analyzing time-frequency domain relationships.

    Supports:
    - Selecting a time window and computing its FFT
    - Selecting a frequency window and reconstructing signal via iFFT
    - Saving updated subplots without altering the original interactive plot

    Attributes:
        ax (2D array of Axes): Grid of matplotlib axes (2x2)
        fig (Figure): The main matplotlib figure
        spectrum (np.ndarray): Full FFT spectrum of the original signal
        flip_spectrum (bool): Whether the spectrum is reversed
        abs_spectrum (bool): Whether to show magnitude or raw FFT values
    """

    def __init__(self, Mx, My, spectrum, ax, fig,
                 line1, line2, line3, line4,
                 vline1, vline2, vline3, vline4,
                 text1, text2, offset, Flip_Sp, Abs_Sp, filepath):
        # Time and frequency axis data from main plots
        self.x1, self.y1 = line1.get_data()
        self.x2, self.y2 = line2.get_data()
        self.x3, self.y3 = line3.get_data()
        self.x4, self.y4 = line4.get_data()

        self.dt = self.x1[1] - self.x1[0]
        self.fs = 1.0 / self.dt

        self.ax = ax
        self.fig = fig

        # Vertical lines and label objects
        self.vline1 = vline1
        self.vline2 = vline2
        self.text1 = text1
        self.vline3 = vline3
        self.vline4 = vline4
        self.text2 = text2

        self.Mx = Mx
        self.My = My
        self.Mt = Mx + 1j * My
        self.offset = offset
        self.Flip_Sp = Flip_Sp
        self.Abs_Sp = Abs_Sp
        self.spectrum = spectrum
        self.filepath = filepath

        # Variables to store interaction coordinates
        self.x1in = self.x1fi = self.x2in = self.x2fi = self.x3in = self.x3fi = self.x4in = self.x4fi = None

    def save_subplot_from_axis(self, ax, filename, outdir="SavedPlots"):
        """
        Saves a clean copy of the specified axis to a PNG file.

        Args:
            ax (matplotlib.axes.Axes): The axis to save.
            filename (str): The name of the file (e.g., 'fft_of_selection.png').
            outdir (str): Output folder to save the files.
        """
        # Ensure output directory exists
        directory = os.path.dirname(self.filepath)
        output_dir = os.path.join(directory, outdir)
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)

        # Create new figure and replicate the subplot
        fig, new_ax = plt.subplots(figsize=(6, 4))

        # Copy lines only with valid x, y
        for line in ax.get_lines():
            x, y = line.get_data()
            if len(x) == len(y):
                new_ax.plot(x, y, linestyle=line.get_linestyle(), color=line.get_color())

        # Copy appearance settings
        new_ax.set_title(ax.get_title())
        new_ax.set_xlabel(ax.get_xlabel())
        new_ax.set_ylabel(ax.get_ylabel())
        new_ax.set_xlim(ax.get_xlim())
        new_ax.set_ylim(ax.get_ylim())
        new_ax.grid(True)

        # Save and clean up
        fig.tight_layout()
        fig.savefig(full_path, bbox_inches='tight', format='svg')
        plt.close(fig)

    def button_press(self, event):
        """
        Captures initial mouse press location in relevant subplot.
        Used for selecting a time or frequency window.
        """
        if event.inaxes is self.ax[0, 0]:
            self.x1in = min(np.searchsorted(self.x1, event.xdata), len(self.x1) - 1)
        elif event.inaxes is self.ax[1, 0]:
            self.x3in = min(np.searchsorted(self.x3, event.xdata), len(self.x3) - 1)
        elif event.inaxes is self.ax[0, 1]:
            self.x2in = event.xdata
            self.vline1.set_xdata([self.x2in])
            plt.draw()
        elif event.inaxes is self.ax[1, 1]:
            self.x4in = event.xdata
            self.vline3.set_xdata([self.x4in])
            plt.draw()

    def button_release(self, event):
        """
        Captures mouse release and handles the following:
        - Computes FFT from selected time range
        - Computes iFFT from selected frequency range
        - Updates the corresponding subplots
        - Saves the subplots to disk (without modifying originals)
        """
        if event.inaxes is self.ax[0, 0]:  # Time domain selection
            self.x1fi = min(np.searchsorted(self.x1, event.xdata), len(self.x1) - 1)

            # Highlight selected time window
            self.ax[0, 0].axvspan(self.x1[self.x1in], self.x1[self.x1fi], color='red', alpha=0.2)

            # Compute FFT of selection
            Spectrum = np.fft.fft(self.Mt[self.x1in:self.x1fi])
            Spectrum = np.fft.fftshift(Spectrum)
            spectrum = Spectrum[::-1] if self.Flip_Sp else Spectrum
            freq = np.linspace(-self.fs / 2, self.fs / 2, spectrum.shape[-1]) + self.offset

            # Replace spectrum subplot lines (after first) with new spectrum
            for line in self.ax[0, 1].lines[1:]:
                line.remove()

            self.ax[0, 1].plot(freq, np.abs(spectrum) if self.Abs_Sp else spectrum, '-', color='red')
            plt.draw()

            # Save plots cleanly
            self.save_subplot_from_axis(self.ax[0, 0], "time_domain_selected.svg")
            self.save_subplot_from_axis(self.ax[0, 1], "fft_of_selection.svg")

        elif event.inaxes is self.ax[1, 0]:  # Frequency range selection
            self.x3fi = min(np.searchsorted(self.x3, event.xdata), len(self.x3) - 1)
            window = np.zeros_like(self.y3)
            window[self.x3in:self.x3fi] = 1.0

            # Highlight selected frequency window
            self.ax[1, 0].axvspan(self.x3[self.x3in], self.x3[self.x3fi], color='red', alpha=0.2)

            # Compute iFFT reconstruction from selected freq range
            Sig = np.fft.ifftshift(self.spectrum * window)
            Sig = np.fft.ifft(Sig)
            sig = Sig[::-1] if self.Flip_Sp else Sig
            t = np.linspace(0, self.dt * len(self.y3), len(self.y3))

            # Update reconstructed signal subplot
            for line in self.ax[1, 1].lines[1:]:
                line.remove()
            self.ax[1, 1].plot(self.x4, self.y4, '-', color='blue')  # Original
            self.ax[1, 1].plot(t, sig.real, '-', color='red')        # Reconstructed
            plt.draw()

            # Save updated frequency and reconstructed signal plots
            self.save_subplot_from_axis(self.ax[1, 0], "selected_freq_range.svg")
            self.save_subplot_from_axis(self.ax[1, 1], "reconstructed_signal.svg")

        elif event.inaxes is self.ax[0, 1]:  # Measuring frequency difference
            self.x2fi = event.xdata
            self.vline2.set_xdata([self.x2fi])
            self.text1.set_text(f'Freq = {abs(self.x2fi - self.x2in):.5f} Hz')
            plt.draw()

        elif event.inaxes is self.ax[1, 1]:  # Measuring time difference
            self.x4fi = event.xdata
            self.vline4.set_xdata([self.x4fi])
            self.text2.set_text(f'Time = {abs(self.x4fi - self.x4in):.5f} s')
            plt.draw()