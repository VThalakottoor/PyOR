"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description: **Testing**
    This module provides the `MaxwellBloch` class for simulating Maxwell-Bloch equations.
"""


from math import sin, cos
from numpy import array, arange, pi, fft
import numpy as np
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rc

class MaxwellBloch:
    def __init__(self,ChemicalShifts,Isochromats):
        """
        Parameter
        """

        self.DTYPE = np.float64

        self.ChemicalShifts = ChemicalShifts # Number of Chemical Shifts
        self.Isochromats = Isochromats # Number of isochromats at each chemical shift.

        # Relaxation
        self.Relaxation_R1 = 0.0 # T1 longitudinal relaxation
        self.Relaxation_R2 = 0.0 # T2 transverse relaxation

        # Chemical Shift value
        self.Omega_X = 0.0
        self.Omega_Y = 0.0
        self.Omega_Z_CS = np.zeros(self.ChemicalShifts, dtype = self.DTYPE)

        # Isichromateds Frequency Bins
        self.FrequencySeparation = 0.0

        # Magnetization at each chemical shifts
        self.Magnetization = np.zeros(self.ChemicalShifts, dtype = self.DTYPE)
        self.M = np.zeros((self.ChemicalShifts,3*self.Isochromats), dtype = self.DTYPE)
        self.Mo = np.zeros((self.ChemicalShifts,self.Isochromats), dtype = self.DTYPE)

        # Flip angle
        self.FlipAngle_Theta = np.zeros(self.ChemicalShifts, dtype = self.DTYPE)
        self.FlipAngle_Phi = np.zeros(self.ChemicalShifts, dtype = self.DTYPE)

        # Radiation Damping
        self.RD_Xi = 0.0
        self.RD_Phase = 0.0

        # B1 Field
        self.B1_Amplitude = 0.0
        self.B1_Frequency = 0.0
        self.B1_Phase = 0.0

        # Acquisition
        self.AQTime = 10.0
        self.DT = 0.0001
        self.ODEMethod = 'DOP853'

        # Plotting
        self.Plot_Xlim = None
        self.Plot_Ylim = None
        self.Plot_Save = False
        self.fig_counter = 1
        self.abs_spectrum = True

    def Initialize(self):
        # Frequency
        self.Omega_X = 2.0 * np.pi * self.Omega_X
        self.Omega_Y = 2.0 * np.pi * self.Omega_Y
        self.Omega_Z_CS = 2.0 * np.pi * self.Omega_Z_CS
        self.FrequencySeparation = 2.0 * np.pi * self.FrequencySeparation

        self.Omega_Z_Band = np.zeros((self.ChemicalShifts,self.Isochromats), dtype=self.DTYPE)

        for i in range(self.ChemicalShifts):
            if self.Isochromats%2 == 0:
                Nhalf = int(self.Isochromats/2)
                self.Omega_Z_Band[i] = np.linspace(self.Omega_Z_CS[i] - Nhalf * self.FrequencySeparation, self.Omega_Z_CS[i] + Nhalf * self.FrequencySeparation, self.Isochromats, endpoint=False, dtype = self.DTYPE)
            else:
                Nhalf = int((self.Isochromats-1)/2)
                self.Omega_Z_Band[i] = np.linspace(self.Omega_Z_CS[i] - Nhalf * self.FrequencySeparation, self.Omega_Z_CS[i] + Nhalf * self.FrequencySeparation, self.Isochromats, endpoint=True, dtype = self.DTYPE)                

        self.Omega_Z = np.reshape(self.Omega_Z_Band,self.ChemicalShifts*self.Isochromats)

        # Equilibrium Magnetization

        self.FlipAngle_Theta = (np.pi/180.0) * self.FlipAngle_Theta
        self.FlipAngle_Phi = (np.pi/180.0) * self.FlipAngle_Phi

        Iso_idx = np.arange(self.Isochromats, dtype=self.DTYPE)
        Iso_center = 0.5 * (self.Isochromats - 1)
        Iso_sigma = self.Isochromats / 6.0
        Iso_base_gauss = np.exp(-0.5 * ((Iso_idx - Iso_center) / Iso_sigma) ** 2)
        Iso_base_gauss = Iso_base_gauss / Iso_base_gauss.sum()   # normalize to sum = 1

        for i in range(self.ChemicalShifts):
            self.Mo[i, :] = self.Magnetization[i] * Iso_base_gauss

        for i in range(self.ChemicalShifts):
            self.M[i, 0::3] = np.absolute(self.Mo[i,:]) * np.sin(self.FlipAngle_Theta[i]) * np.cos(self.FlipAngle_Phi[i])
            self.M[i, 1::3] = np.absolute(self.Mo[i,:]) * np.sin(self.FlipAngle_Theta[i]) * np.sin(self.FlipAngle_Phi[i])
            self.M[i, 2::3] = np.absolute(self.Mo[i,:]) * np.cos(self.FlipAngle_Theta[i])

        tol = 1e-16
        self.M[np.abs(self.M) < tol] = 0.0 # zero anything smaller than tol.

        self.M_Band = self.M
        self.Mo_Band = self.Mo

        self.M = np.reshape(self.M, 3 * self.Isochromats * self.ChemicalShifts)
        self.Mo = np.reshape(self.Mo, self.Isochromats * self.ChemicalShifts)

        # Simulation points
        self.AQPoints = int(self.AQTime/self.DT)
        self.FS = 1.0/self.DT
        self.tpoints = np.linspace(0.0, self.AQTime, self.AQPoints, endpoint=True)

        # Radiation Damping
        self.RD_Phase = (np.pi/180.0) * self.RD_Phase       

        # B1 Field
        self.B1_Amplitude = 2.0 * np.pi * self.B1_Amplitude
        self.B1_Frequency = 2.0 * np.pi * self.B1_Frequency
        self.B1_Phase = (np.pi/180.0) * self.B1_Phase

    def Evolution(self):

        M = self.M
        Mo = self.Mo
        Isochromats = self.Isochromats
        ChemicalShifts = self.ChemicalShifts
        RD_Xi = self.RD_Xi
        RD_Phase = self.RD_Phase
        Omega_X = self.Omega_X
        Omega_Y = self.Omega_Y
        Omega_Z = self.Omega_Z
        R1 = self.Relaxation_R1
        R2 = self.Relaxation_R2
        B1_Amplitude = self.B1_Amplitude
        B1_Frequency = self.B1_Frequency
        B1_Phase = self.B1_Phase

        def MDOT(t,M,Isochromats,ChemicalShifts,RD_Xi,RD_Phase,Omega_X,Omega_Y,Omega_Z,R1,R2,B1_Amplitude,B1_Frequency,B1_Phase):

            Mx = M[0::3]
            My = M[1::3]
            Mz = M[2::3]

            Omega_RD = np.zeros((Isochromats * ChemicalShifts))
            omega_RD = 1j * RD_Xi * (np.average(Mx) + 1j * np.average(My)) * np.exp(-1j * RD_Phase)

            B1_Field = B1_Amplitude * np.exp(1j * (B1_Frequency * t + B1_Phase)) 

            Wx = Omega_X + omega_RD.real + B1_Field.real
            Wy = Omega_Y + omega_RD.imag + B1_Field.imag
            Wz = Omega_Z

            # Equation 13 of https://doi.org/10.1063/1.470468
            Mdot = np.zeros((3 * Isochromats * ChemicalShifts))

            Mdot[0::3] = -R2 * Mx - Wz * My - Wy * Mz
            Mdot[1::3] = Wz * Mx - R2 * My + Wx * Mz
            Mdot[2::3] = Wy * Mx - Wx * My - R1 * Mz + R1 * Mo

            return Mdot

        start_time = time.time()
        Msol = solve_ivp(MDOT,[0,self.AQTime],M,method=self.ODEMethod,t_eval=self.tpoints,args=(Isochromats,ChemicalShifts,RD_Xi,RD_Phase,Omega_X,Omega_Y,Omega_Z,R1,R2,B1_Amplitude,B1_Frequency,B1_Phase),atol = 1e-10, rtol = 1e-10)
        end_time = time.time()
        timetaken = end_time - start_time
        print(f"Total time = {timetaken}")

        self.tpoints, self.Mpoints = Msol.t, Msol.y
        self.Mx = np.sum(self.Mpoints[0::3,:], axis=0) # Adding all Mx components of all spins
        self.My = np.sum(self.Mpoints[1::3,:], axis=0) # Adding all My components of all spins
        self.Mz = np.sum(self.Mpoints[2::3,:], axis=0) # Adding all Mz components of all spins
        self.Mabs = np.sqrt(self.Mx**2 + self.My**2)   
        self.Signal = self.Mx + 1j * self.My
        self.DT = self.tpoints[1] - self.tpoints[0]
        self.FS = 1.0/self.DT

        Spectrum = np.fft.fft(self.Signal)
        self.Spectrum = np.fft.fftshift(Spectrum)
        self.Freq = np.linspace(-self.FS/2,self.FS/2,self.Signal.shape[-1])

        print("Simulation is completed")

    def Ploting_MxMyMz(self):

        rc('font', weight='bold')
        fig = plt.figure(self.fig_counter,constrained_layout=True, figsize=(15, 5))
        spec = fig.add_gridspec(1, 1)
        self.fig_counter += 1
        ax1 = fig.add_subplot(spec[0, 0])

        ax1.plot(self.tpoints,self.Mx,linewidth=3.0,color='blue',label = "Mx")
        ax1.plot(self.tpoints,self.My,linewidth=3.0,color='green',label = "My")

        ax1.set_xlabel(r'Time (s)', fontsize=25, color='black',fontweight='bold')
        ax1.set_ylabel(r'$M_{T}$ (AU)', fontsize=25, color='blue',fontweight='bold')
        ax1.legend(fontsize=25,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        ax1.set_xlim(self.Plot_Xlim)
        ax1.set_ylim(self.Plot_Ylim)
        #ax1.text(0.05, 200000, '(a)', ha='center', fontsize=25, color='black',fontweight='bold')

        ax10 = ax1.twinx()
        ax10.plot(self.tpoints,self.Mz,linewidth=3.0,color='red',label = "Mz")
        ax10.set_xlabel(r'Time (s)', fontsize=30, color='black',fontweight='bold')
        ax10.set_ylabel(r'$M_{Z}$ (AU)', fontsize=30, color='red',fontweight='bold')
        ax10.legend(fontsize=30,frameon=False)
        ax10.tick_params(axis='both',labelsize=20)
        ax1.set_xlim(self.Plot_Xlim)
        ax1.set_ylim(self.Plot_Ylim)
        if self.Plot_Save:
            plt.savefig('MxMyMz.pdf',bbox_inches='tight')        
        
    def Ploting_Spectrum(self):
        fig = plt.figure(self.fig_counter,constrained_layout=True, figsize=(15, 5))
        spec = fig.add_gridspec(1, 1)
        self.fig_counter += 1

        ax1 = fig.add_subplot(spec[0, 0])

        ax1.plot(self.Freq,self.Spectrum,linewidth=3.0,color='black')
        ax1.set_xlabel(r'Frequency (Hz)', fontsize=25, color='green',fontweight='bold')
        ax1.set_ylabel(r'Spectrum (AU)', fontsize=25, color='black',fontweight='bold')
        #ax1.legend(fontsize=25,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        ax1.set_xlim(self.Plot_Xlim)
        ax1.set_ylim(self.Plot_Ylim)
        if self.Plot_Save:
            plt.savefig('Spectrum.pdf',bbox_inches='tight')     

    def Plotting_Sphere(self):
        S_phi = np.linspace(0, np.pi, 20)
        S_theta = np.linspace(0, 2*np.pi, 20)
        S_phi, S_theta = np.meshgrid(S_phi, S_theta)
        S_x = np.sum(self.Magnetization) * np.sin(S_phi) * np.cos(S_theta)
        S_y = np.sum(self.Magnetization) * np.sin(S_phi) * np.sin(S_theta)
        S_z = np.sum(self.Magnetization) * np.cos(S_phi)

        tlim1 = 0 #-10000
        tlim2 = -1
        ax = plt.figure(self.fig_counter,figsize=(10,10)).add_subplot(projection='3d')
        self.fig_counter += 1
        ax.plot_wireframe(S_x,S_y,S_z, color="cyan",linewidth=1.0)

        ax.plot(self.Mx[tlim1:tlim2],self.My[tlim1:tlim2],self.Mz[tlim1:tlim2], color="black",linewidth=1.0)
        #ax.plot(Mpoints[0,tlim1:tlim2],Mpoints[1,tlim1:tlim2],Mpoints[2,tlim1:tlim2], color="green",linewidth=1.0) 
        #ax.plot(Mpoints[3,tlim1:tlim2],Mpoints[4,tlim1:tlim2],Mpoints[5,tlim1:tlim2], color="blue",linewidth=1.0) 
        ax.view_init(10, 20)
        ax.set_xlabel(r'My', fontsize=14, color='black',fontweight='bold')
        ax.set_ylabel(r'Mx', fontsize=14, color='black',fontweight='bold')
        ax.set_zlabel(r'Mz', fontsize=14, color='black',fontweight='bold')
        ax.tick_params(axis='both',labelsize=10)
        ax.grid(True, linestyle='-.')
        if self.Plot_Save:
            plt.savefig('Sphere.pdf',bbox_inches='tight') 

        plt.show() 

    def Plotting_FourierAnalyzer(self):
        self.Setup_Plot()
        self.Connect_Events()       

    def Setup_Plot(self):
        """Creates and configures a 2x2 subplot grid."""
        self.figsize = (12, 9)
        self.fig = plt.figure(self.fig_counter, figsize=self.figsize)
        self.ax = self.fig.subplots(2, 2)
        self.fig_counter += 1

        # Time domain
        self.line1, = self.ax[0, 0].plot(self.tpoints, self.Signal.real, '-', color='green')
        self.ax[0, 0].set_title("Time Domain")
        self.ax[0, 0].set_xlabel("Time [s]")
        self.ax[0, 0].set_ylabel("Signal")
        self.ax[0, 0].grid()

        # Frequency domain (top)
        self.vline1 = self.ax[0, 1].axvline(color='k', lw=0.8, ls='--')
        self.vline2 = self.ax[0, 1].axvline(color='k', lw=0.8, ls='--')
        self.text1 = self.ax[0, 1].text(0.0, 0.95, '', transform=self.ax[0, 1].transAxes)
        spectrum_data = np.abs(self.Spectrum) if self.abs_spectrum else self.Spectrum
        self.line2, = self.ax[0, 1].plot(self.Freq, spectrum_data, '-', color='green')
        self.ax[0, 1].set_title("Frequency Domain (Top)")
        self.ax[0, 1].set_xlabel("Frequency [Hz]")
        self.ax[0, 1].set_ylabel("Spectrum")
        if self.Plot_Xlim is not None:
            self.ax[0, 1].set_xlim(self.Plot_Xlim)
        self.ax[0, 1].grid()

        # Frequency domain (bottom)
        self.line3, = self.ax[1, 0].plot(self.Freq, spectrum_data, '-', color='green')
        self.ax[1, 0].set_title("Frequency Domain (Bottom)")
        self.ax[1, 0].set_xlabel("Frequency [Hz]")
        self.ax[1, 0].set_ylabel("Spectrum")
        if self.Plot_Xlim is not None:
            self.ax[1, 0].set_xlim(self.Plot_Xlim)
        self.ax[1, 0].grid()

        # Reconstructed signal
        self.vline3 = self.ax[1, 1].axvline(color='k', lw=0.8, ls='--')
        self.vline4 = self.ax[1, 1].axvline(color='k', lw=0.8, ls='--')
        self.text2 = self.ax[1, 1].text(0.0, 0.95, '', transform=self.ax[1, 1].transAxes)
        self.line4, = self.ax[1, 1].plot(self.tpoints, self.Signal.real, '-', color='green')
        self.ax[1, 1].set_title("Reconstructed Signal")
        self.ax[1, 1].set_xlabel("Time [s]")
        self.ax[1, 1].set_ylabel("Signal")
        self.ax[1, 1].grid()

    def Connect_Events(self):
        """Binds mouse events for interaction."""
        self.fourier = Fourier(self.Mx, self.My, self.Spectrum, self.ax, self.fig,
                               self.line1, self.line2, self.line3, self.line4,
                               self.vline1, self.vline2, self.vline3, self.vline4,
                               self.text1, self.text2, self.abs_spectrum)

        self.fig.canvas.mpl_connect("button_press_event", self.fourier.button_press)
        self.fig.canvas.mpl_connect("button_release_event", self.fourier.button_release)

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
        abs_spectrum (bool): Whether to show magnitude or raw FFT values
    """

    def __init__(self, Mx, My, spectrum, ax, fig,
                 line1, line2, line3, line4,
                 vline1, vline2, vline3, vline4,
                 text1, text2, Abs_Sp):
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
        self.Abs_Sp = Abs_Sp
        self.spectrum = spectrum

        # Variables to store interaction coordinates
        self.x1in = self.x1fi = self.x2in = self.x2fi = self.x3in = self.x3fi = self.x4in = self.x4fi = None


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
            spectrum = Spectrum
            freq = np.linspace(-self.fs / 2, self.fs / 2, spectrum.shape[-1])

            # Replace spectrum subplot lines (after first) with new spectrum
            for line in self.ax[0, 1].lines[1:]:
                line.remove()

            self.ax[0, 1].plot(freq, np.abs(spectrum) if self.Abs_Sp else spectrum, '-', color='red')
            plt.draw()

        elif event.inaxes is self.ax[1, 0]:  # Frequency range selection
            self.x3fi = min(np.searchsorted(self.x3, event.xdata), len(self.x3) - 1)
            window = np.zeros_like(self.y3)
            window[self.x3in:self.x3fi] = 1.0

            # Highlight selected frequency window
            self.ax[1, 0].axvspan(self.x3[self.x3in], self.x3[self.x3fi], color='red', alpha=0.2)

            # Compute iFFT reconstruction from selected freq range
            Sig = np.fft.ifftshift(self.spectrum * window)
            Sig = np.fft.ifft(Sig)
            sig = Sig
            t = np.linspace(0, self.dt * len(self.y3), len(self.y3))

            # Update reconstructed signal subplot
            for line in self.ax[1, 1].lines[1:]:
                line.remove()
            self.ax[1, 1].plot(self.x4, self.y4, '-', color='blue')  # Original
            self.ax[1, 1].plot(t, sig.real, '-', color='red')        # Reconstructed
            plt.draw()

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