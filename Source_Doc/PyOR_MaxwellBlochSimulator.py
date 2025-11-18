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
        self.FlipAngle_Theta = 0.0
        self.FlipAngle_Phi = 0.0

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
            self.Mo[i, :] = self.Magnetization[i] #* Iso_base_gauss

        for i in range(self.ChemicalShifts):
            self.M[i, 0::3] = np.absolute(self.Mo[i,:]) * np.sin(self.FlipAngle_Theta) * np.cos(self.FlipAngle_Phi)
            self.M[i, 1::3] = np.absolute(self.Mo[i,:]) * np.sin(self.FlipAngle_Theta) * np.sin(self.FlipAngle_Phi)
            self.M[i, 2::3] = np.absolute(self.Mo[i,:]) * np.cos(self.FlipAngle_Theta)

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

 