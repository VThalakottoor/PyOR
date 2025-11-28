"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    Maxwell-Bloch with FFT dipolar field (PCCP-style) + optional JAX backend
    + static Grandient in Omega_Z along X, Y, Z lattice directions.
"""

from math import sin, cos
import time
import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rc

# --- Optional JAX + Diffrax backend -----------------------------------------
try:
    import jax
    import jax.numpy as jnp
    import diffrax as dfx
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def MDOT_Jax(t, M, args):
    """
    JAX-compatible right-hand side for Maxwell–Bloch with
    FFT dipolar field + uniform demag term + mean dipole field.

    Parameters
    ----------
    t : float
        Time (scalar).
    M : jnp.ndarray, shape (3 * Nspin,)
        State vector: [Mx0, My0, Mz0, Mx1, My1, Mz1, ...].
    args : tuple
        (Isochromats, ChemicalShifts,
         RD_Xi, RD_Phase,
         Omega_X, Omega_Y, Omega_Z,
         R1, R2,
         B1_Amplitude, B1_Frequency, B1_Phase,
         Mo_flat,
         Dipolar_On, Demag_Coefficient,
         Mean_Dipolar_On, Mean_Dipolar_Strength,
         Lattice_Nx, Lattice_Ny, Lattice_Nz,
         Padd_X, Padd_Y, Padd_Z,
         Gamma, Permeability,
         Mask_flat, KTensor_z2)
    """
    (Isochromats, ChemicalShifts,
     RD_Xi, RD_Phase,
     Omega_X, Omega_Y, Omega_Z,
     R1, R2,
     B1_Amplitude, B1_Frequency, B1_Phase,
     Mo_flat,
     Dipolar_On, Demag_Coefficient,
     Mean_Dipolar_On, Mean_Dipolar_Strength,
     Lattice_Nx, Lattice_Ny, Lattice_Nz,
     Padd_X, Padd_Y, Padd_Z,
     Gamma, Permeability,
     Mask_flat, KTensor_z2) = args

    Nspin = Isochromats * ChemicalShifts
    Mx = M[0::3]
    My = M[1::3]
    Mz = M[2::3]

    # Radiation damping (macroscopic)
    mx_avg = jnp.mean(Mx)
    my_avg = jnp.mean(My)
    omega_RD = 1j * RD_Xi * (mx_avg + 1j * my_avg) * jnp.exp(-1j * RD_Phase)

    # RF field
    B1_Field = B1_Amplitude * jnp.exp(1j * (B1_Frequency * t + B1_Phase))

    # --- Dipolar field via FFT (PCCP style) + demag ------------------------
    Wx_dp = jnp.zeros_like(Mx)
    Wy_dp = jnp.zeros_like(My)
    Wz_dp = jnp.zeros_like(Mz)

    if Dipolar_On != 0:
        Nx = Lattice_Nx
        Ny = Lattice_Ny
        Nz = Lattice_Nz
        px = Padd_X
        py = Padd_Y
        pz = Padd_Z

        # reshape to 3D lattice
        Mx_grid = jnp.reshape(Mx, (Nx, Ny, Nz))
        My_grid = jnp.reshape(My, (Nx, Ny, Nz))
        Mz_grid = jnp.reshape(Mz, (Nx, Ny, Nz))

        # zero padding
        pad_cfg = ((0, px), (0, py), (0, pz))
        Mx_p = jnp.pad(Mx_grid, pad_cfg, mode="constant", constant_values=0.0)
        My_p = jnp.pad(My_grid, pad_cfg, mode="constant", constant_values=0.0)
        Mz_p = jnp.pad(Mz_grid, pad_cfg, mode="constant", constant_values=0.0)

        # FFT
        Mx_k = jnp.fft.fftn(Mx_p)
        My_k = jnp.fft.fftn(My_p)
        Mz_k = jnp.fft.fftn(Mz_p)
        Mx_k = jnp.fft.fftshift(Mx_k)
        My_k = jnp.fft.fftshift(My_k)
        Mz_k = jnp.fft.fftshift(Mz_k)

        # dipolar kernel (same as PCCP_program: (1/6)(1 - 3 kz^2), (2/6)(3 kz^2 - 1))
        Kz2 = KTensor_z2
        Cx = (1.0 / 6.0) * (1.0 - 3.0 * Kz2) * Permeability
        Cy = (1.0 / 6.0) * (1.0 - 3.0 * Kz2) * Permeability
        Cz = (2.0 / 6.0) * (3.0 * Kz2 - 1.0) * Permeability

        Mx_k = Cx * Mx_k
        My_k = Cy * My_k
        Mz_k = Cz * Mz_k

        Mx_k = jnp.fft.ifftshift(Mx_k)
        My_k = jnp.fft.ifftshift(My_k)
        Mz_k = jnp.fft.ifftshift(Mz_k)

        Mx_d = jnp.fft.ifftn(Mx_k).real
        My_d = jnp.fft.ifftn(My_k).real
        Mz_d = jnp.fft.ifftn(Mz_k).real

        Mx_d = Mx_d[0:Nx, 0:Ny, 0:Nz]
        My_d = My_d[0:Nx, 0:Ny, 0:Nz]
        Mz_d = Mz_d[0:Nx, 0:Ny, 0:Nz]

        Bx_flat = jnp.reshape(Mx_d, (Nspin,))
        By_flat = jnp.reshape(My_d, (Nspin,))
        Bz_flat = jnp.reshape(Mz_d, (Nspin,))

        Wx_dp = -Gamma * Bx_flat * Mask_flat
        Wy_dp = -Gamma * By_flat * Mask_flat
        Wz_dp = -Gamma * Bz_flat * Mask_flat

        # demag-like uniform term with user-defined coefficient
        Mx_avg_dp = jnp.mean(Mx)
        My_avg_dp = jnp.mean(My)
        Mz_avg_dp = jnp.mean(Mz)

        factor = -Gamma * Permeability * Demag_Coefficient

        Wx_dp = Wx_dp + factor * (-Mx_avg_dp)
        Wy_dp = Wy_dp + factor * (-My_avg_dp)
        Wz_dp = Wz_dp + factor * (2.0 * Mz_avg_dp)

    # --- Mean dipole field (independent of FFT dipole) ----------------------
    Wx_mean = jnp.zeros_like(Mx)
    Wy_mean = jnp.zeros_like(My)
    Wz_mean = jnp.zeros_like(Mz)
    if (Mean_Dipolar_On != 0) and (Mean_Dipolar_Strength != 0.0):
        Mz_avg_mean = jnp.mean(Mz)
        Wz_mean = Mean_Dipolar_Strength * Mz_avg_mean * jnp.ones_like(Mz)

    # total effective fields
    Wx = Omega_X + omega_RD.real + B1_Field.real + Wx_dp + Wx_mean
    Wy = Omega_Y + omega_RD.imag + B1_Field.imag + Wy_dp + Wy_mean
    Wz = Omega_Z + Wz_dp + Wz_mean

    Mdot = jnp.zeros_like(M)
    Mdot = Mdot.at[0::3].set(-R2 * Mx - Wz * My - Wy * Mz)
    Mdot = Mdot.at[1::3].set(Wz * Mx - R2 * My + Wx * Mz)
    Mdot = Mdot.at[2::3].set(Wy * Mx - Wx * My - R1 * Mz + R1 * Mo_flat)
    return Mdot


class MaxwellBloch:
    def __init__(self, ChemicalShifts, Isochromats):
        self.DTYPE = np.float64

        self.ChemicalShifts = ChemicalShifts
        self.Isochromats = Isochromats

        # Relaxation
        self.Relaxation_R1 = 0.0
        self.Relaxation_R2 = 0.0

        # Chemical shifts
        self.Omega_X = 0.0
        self.Omega_Y = 0.0
        self.Omega_Z_CS = np.zeros(self.ChemicalShifts, dtype=self.DTYPE)
        self.FrequencySeparation = 0.0

        # Magnetization per chemical shift
        self.Magnetization = np.zeros(self.ChemicalShifts, dtype=self.DTYPE)
        self.M = np.zeros((self.ChemicalShifts, 3 * self.Isochromats), dtype=self.DTYPE)
        self.Mo = np.zeros((self.ChemicalShifts, self.Isochromats), dtype=self.DTYPE)

        # Flip angles
        self.FlipAngle_Theta = np.zeros(self.ChemicalShifts, dtype=self.DTYPE)
        self.FlipAngle_Phi = np.zeros(self.ChemicalShifts, dtype=self.DTYPE)

        # Radiation damping
        self.RD_Xi = 0.0
        self.RD_Phase = 0.0

        # B1 RF field
        self.B1_Amplitude = 0.0
        self.B1_Frequency = 0.0
        self.B1_Phase = 0.0

        # Acquisition
        self.AQTime = 10.0
        self.DT = 0.0001
        self.tpointsFull = None
        self.SignalFull = None
        self.ODEMethod = "DOP853"
        self.ODE_Backend = "scipy"  # "scipy" or "jax"
        self.ODE_Stiff = False
        self.JAX_ODEMethod = "tsit5"
        self.JAX_Device = "cpu"
        self.JAX_MaxSteps = 2_000_000

        # Dipolar field parameters (FFT + demag, PCCP-style)
        self.Dipolar_On = False
        # Generic demag coefficient: factor = -Gamma * mu0 * Demag_Coefficient
        self.Demag_Coefficient = 0.0
        self.Permeability = 4.0 * np.pi * 1.0e-7
        self.Gamma = 2.675e8

        # Mean dipole (independent of FFT dipole)
        # Effective field: Wz_mean = Mean_Dipolar_Strength * <Mz>
        self.Mean_Dipolar_On = False
        self.Mean_Dipolar_Strength = 0.0

        # 3D lattice
        self.Lattice_Nx = 1
        self.Lattice_Ny = 1
        self.Lattice_Nz = self.ChemicalShifts * self.Isochromats
        self.Padd_X = 0
        self.Padd_Y = 0
        self.Padd_Z = 0
        self.Mask3D = None
        self.Mask = None
        self.KTensor = None  # last index: [0,1,2]; we use [:,:,:,2]

        # ---- Grandient in Omega_Z (frequency bins between lattice layers) ----
        # Grandient_* are in Hz per lattice step along each axis.
        self.Grandient_On = False
        self.Grandient_X = 0.0   # Hz per lattice step along x
        self.Grandient_Y = 0.0   # Hz per lattice step along y
        self.Grandient_Z = 0.0   # Hz per lattice step along z
        # Omega_Z contribution from Grandient in rad/s (flattened, length Nspin)
        self.Grandient_OmegaZ = None

        # Shape of the sample inside lattice: "full", "sphere", "cylinder"
        self.Lattice_Shape = "full"

        # Plotting
        self.Plot_Xlim = None
        self.Plot_Ylim = None
        self.Plot_Save = False
        self.fig_counter = 1
        self.abs_spectrum = True

    def BuildShapeMask(self):
        Nx = int(self.Lattice_Nx)
        Ny = int(self.Lattice_Ny)
        Nz = int(self.Lattice_Nz)
        Nspin = self.ChemicalShifts * self.Isochromats
        if Nx * Ny * Nz != Nspin:
            raise ValueError("Lattice_Nx * Lattice_Ny * Lattice_Nz must equal ChemicalShifts * Isochromats")
        if self.Mask3D is not None:
            return

        shape = str(self.Lattice_Shape).lower()

        # Coordinates centered around zero
        x = np.arange(Nx, dtype=self.DTYPE) - 0.5 * (Nx - 1)
        y = np.arange(Ny, dtype=self.DTYPE) - 0.5 * (Ny - 1)
        z = np.arange(Nz, dtype=self.DTYPE) - 0.5 * (Nz - 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        if shape == "sphere":
            a = 0.5 * (Nx - 1) if Nx > 1 else 0.5
            b = 0.5 * (Ny - 1) if Ny > 1 else 0.5
            c = 0.5 * (Nz - 1) if Nz > 1 else 0.5
            a = max(a, 0.5)
            b = max(b, 0.5)
            c = max(c, 0.5)
            maskc = (X**2 / (a**2)) + (Y**2 / (b**2)) + (Z**2 / (c**2)) <= 1.0
            self.Mask3D = maskc.astype(self.DTYPE)
        elif shape == "cylinder":
            a = 0.5 * (Nx - 1) if Nx > 1 else 0.5
            b = 0.5 * (Ny - 1) if Ny > 1 else 0.5
            a = max(a, 0.5)
            b = max(b, 0.5)
            maskc = (X**2 / (a**2)) + (Y**2 / (b**2)) <= 1.0
            self.Mask3D = maskc.astype(self.DTYPE)
        else:
            self.Mask3D = np.ones((Nx, Ny, Nz), dtype=self.DTYPE)

        self.Mask = self.Mask3D.reshape(Nspin)

    def BuildKSpaceLattice(self):
        if not self.Dipolar_On:
            return
        if self.KTensor is not None:
            return
        Nx = int(self.Lattice_Nx)
        Ny = int(self.Lattice_Ny)
        Nz = int(self.Lattice_Nz)
        px = int(self.Padd_X)
        py = int(self.Padd_Y)
        pz = int(self.Padd_Z)
        kx = np.linspace(-0.5, 0.5, Nx + px, endpoint=True, dtype=self.DTYPE)
        ky = np.linspace(-0.5, 0.5, Ny + py, endpoint=True, dtype=self.DTYPE)
        kz = np.linspace(-0.5, 0.5, Nz + pz, endpoint=True, dtype=self.DTYPE)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
        eps = 1.0e-15
        Kmag = np.where(Kmag < eps, 1.0, Kmag)
        KZ_unit = KZ / Kmag
        KTensor = np.zeros(KX.shape + (3,), dtype=self.DTYPE)
        KTensor[:, :, :, 2] = KZ_unit * KZ_unit
        self.KTensor = KTensor

    def BuildGradientOmegaZ(self):
        """
        Build static Grandient contribution to Omega_Z (in rad/s).

        Grandient_X, Grandient_Y, Grandient_Z are specified in Hz per
        lattice step along x, y, z. Lattice coordinates are centered
        around zero, same convention as in BuildShapeMask().
        """
        Nspin = self.ChemicalShifts * self.Isochromats

        # If gradient is off or zero, store a zero field
        if (not self.Grandient_On) or (
            self.Grandient_X == 0.0 and
            self.Grandient_Y == 0.0 and
            self.Grandient_Z == 0.0
        ):
            self.Grandient_OmegaZ = np.zeros(Nspin, dtype=self.DTYPE)
            return

        Nx = int(self.Lattice_Nx)
        Ny = int(self.Lattice_Ny)
        Nz = int(self.Lattice_Nz)

        if Nx * Ny * Nz != Nspin:
            raise ValueError(
                "For Grandient: Lattice_Nx * Lattice_Ny * Lattice_Nz must "
                "equal ChemicalShifts * Isochromats"
            )

        # Lattice coordinates centered around 0 (same as BuildShapeMask)
        x = np.arange(Nx, dtype=self.DTYPE) - 0.5 * (Nx - 1)
        y = np.arange(Ny, dtype=self.DTYPE) - 0.5 * (Ny - 1)
        z = np.arange(Nz, dtype=self.DTYPE) - 0.5 * (Nz - 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Grandient_* are in Hz per step -> convert to rad/s per step
        gx = 2.0 * np.pi * self.Grandient_X
        gy = 2.0 * np.pi * self.Grandient_Y
        gz = 2.0 * np.pi * self.Grandient_Z

        # Linear gradient field in rad/s
        dOmega = gx * X + gy * Y + gz * Z  # shape (Nx, Ny, Nz)
        self.Grandient_OmegaZ = dOmega.reshape(Nspin)

    def Initialize(self):
        self.Omega_X = 2.0 * np.pi * self.Omega_X
        self.Omega_Y = 2.0 * np.pi * self.Omega_Y
        self.Omega_Z_CS = 2.0 * np.pi * self.Omega_Z_CS
        self.FrequencySeparation = 2.0 * np.pi * self.FrequencySeparation

        self.Omega_Z_Band = np.zeros((self.ChemicalShifts, self.Isochromats), dtype=self.DTYPE)
        for i in range(self.ChemicalShifts):
            if self.Isochromats % 2 == 0:
                Nhalf = int(self.Isochromats / 2)
                self.Omega_Z_Band[i] = np.linspace(
                    self.Omega_Z_CS[i] - Nhalf * self.FrequencySeparation,
                    self.Omega_Z_CS[i] + Nhalf * self.FrequencySeparation,
                    self.Isochromats,
                    endpoint=False,
                    dtype=self.DTYPE,
                )
            else:
                Nhalf = int((self.Isochromats - 1) / 2)
                self.Omega_Z_Band[i] = np.linspace(
                    self.Omega_Z_CS[i] - Nhalf * self.FrequencySeparation,
                    self.Omega_Z_CS[i] + Nhalf * self.FrequencySeparation,
                    self.Isochromats,
                    endpoint=True,
                    dtype=self.DTYPE,
                )

        # Base Omega_Z from chemical shifts + inhomogeneous band (rad/s)
        self.Omega_Z = np.reshape(self.Omega_Z_Band, self.ChemicalShifts * self.Isochromats)

        # Flip angles in radians
        self.FlipAngle_Theta = (np.pi / 180.0) * self.FlipAngle_Theta
        self.FlipAngle_Phi = (np.pi / 180.0) * self.FlipAngle_Phi

        # Gaussian distribution over isochromats
        Iso_idx = np.arange(self.Isochromats, dtype=self.DTYPE)
        Iso_center = 0.5 * (self.Isochromats - 1)
        Iso_sigma = self.Isochromats / 6.0
        Iso_base_gauss = np.exp(-0.5 * ((Iso_idx - Iso_center) / Iso_sigma) ** 2)
        Iso_base_gauss = Iso_base_gauss / Iso_base_gauss.sum()

        for i in range(self.ChemicalShifts):
            self.Mo[i, :] = self.Magnetization[i] * Iso_base_gauss

        for i in range(self.ChemicalShifts):
            self.M[i, 0::3] = np.abs(self.Mo[i, :]) * np.sin(self.FlipAngle_Theta[i]) * np.cos(self.FlipAngle_Phi[i])
            self.M[i, 1::3] = np.abs(self.Mo[i, :]) * np.sin(self.FlipAngle_Theta[i]) * np.sin(self.FlipAngle_Phi[i])
            self.M[i, 2::3] = np.abs(self.Mo[i, :]) * np.cos(self.FlipAngle_Theta[i])

        tol = 1.0e-16
        self.M[np.abs(self.M) < tol] = 0.0

        self.M_Band = self.M.copy()
        self.Mo_Band = self.Mo.copy()

        self.M = np.reshape(self.M, 3 * self.Isochromats * self.ChemicalShifts)
        self.Mo = np.reshape(self.Mo, self.Isochromats * self.ChemicalShifts)

        self.AQPoints = int(self.AQTime / self.DT)
        self.FS = 1.0 / self.DT
        self.tpoints = np.linspace(0.0, self.AQTime, self.AQPoints, endpoint=True)

        self.RD_Phase = (np.pi / 180.0) * self.RD_Phase
        self.B1_Amplitude = 2.0 * np.pi * self.B1_Amplitude
        self.B1_Frequency = 2.0 * np.pi * self.B1_Frequency
        self.B1_Phase = (np.pi / 180.0) * self.B1_Phase

        self.BuildShapeMask()
        self.BuildKSpaceLattice()
        self.BuildGradientOmegaZ()

        # Add Grandient contribution (already in rad/s) to Omega_Z
        if self.Grandient_OmegaZ is not None:
            self.Omega_Z = self.Omega_Z + self.Grandient_OmegaZ

    def Update(self):
        self.AQPoints = int(self.AQTime / self.DT)
        self.FS = 1.0 / self.DT
        self.tpoints = np.linspace(0.0, self.AQTime, self.AQPoints, endpoint=True)

    def ApplyInstantPulse(self, angle_deg, axis='x'):
        """
        Apply an instantaneous hard pulse: rotate all spins by angle_deg
        around axis 'x', 'y', or 'z'. Works on self.M (flattened).
        """
        theta = np.deg2rad(angle_deg)

        # M is shape (3*Nspin,) = [Mx0,My0,Mz0,Mx1,My1,Mz1,...]
        Mx = self.M[0::3].copy()
        My = self.M[1::3].copy()
        Mz = self.M[2::3].copy()

        axis = axis.lower()
        if axis == 'x':
            # rotation around x: (Mx, My, Mz) -> (Mx, My cosθ - Mz sinθ, My sinθ + Mz cosθ)
            My_new = My * np.cos(theta) - Mz * np.sin(theta)
            Mz_new = My * np.sin(theta) + Mz * np.cos(theta)
            Mx_new = Mx
        elif axis == 'y':
            # rotation around y: (Mx, My, Mz) -> (Mx cosθ + Mz sinθ, My, -Mx sinθ + Mz cosθ)
            Mx_new = Mx * np.cos(theta) + Mz * np.sin(theta)
            Mz_new = -Mx * np.sin(theta) + Mz * np.cos(theta)
            My_new = My
        elif axis == 'z':
            # rotation around z: (Mx, My, Mz) -> (Mx cosθ - My sinθ, Mx sinθ + My cosθ, Mz)
            Mx_new = Mx * np.cos(theta) - My * np.sin(theta)
            My_new = Mx * np.sin(theta) + My * np.cos(theta)
            Mz_new = Mz
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        # write back into flattened M
        self.M[0::3] = Mx_new
        self.M[1::3] = My_new
        self.M[2::3] = Mz_new
       
    def DipolarFieldScipy(self, Mvec):
        """
        FFT dipolar field + uniform demag term, SciPy version.
        """
        if not self.Dipolar_On:
            Nspin = self.ChemicalShifts * self.Isochromats
            zeros = np.zeros(Nspin, dtype=self.DTYPE)
            return zeros, zeros, zeros

        Nx = int(self.Lattice_Nx)
        Ny = int(self.Lattice_Ny)
        Nz = int(self.Lattice_Nz)
        px = int(self.Padd_X)
        py = int(self.Padd_Y)
        pz = int(self.Padd_Z)
        mask = self.Mask
        K = self.KTensor
        Gamma = self.Gamma
        mu0 = self.Permeability

        Nspin = Nx * Ny * Nz
        Mx_flat = Mvec[0::3]
        My_flat = Mvec[1::3]
        Mz_flat = Mvec[2::3]

        M4 = np.zeros((Nx, Ny, Nz, 3), dtype=self.DTYPE)
        M4[:, :, :, 0] = Mx_flat.reshape(Nx, Ny, Nz)
        M4[:, :, :, 1] = My_flat.reshape(Nx, Ny, Nz)
        M4[:, :, :, 2] = Mz_flat.reshape(Nx, Ny, Nz)

        Mx = M4[:, :, :, 0]
        My = M4[:, :, :, 1]
        Mz = M4[:, :, :, 2]

        if px > 0 or py > 0 or pz > 0:
            Mx = np.pad(Mx, ((0, px), (0, py), (0, pz)), mode="constant", constant_values=0.0)
            My = np.pad(My, ((0, px), (0, py), (0, pz)), mode="constant", constant_values=0.0)
            Mz = np.pad(Mz, ((0, px), (0, py), (0, pz)), mode="constant", constant_values=0.0)

        Mx_k = np.fft.fftn(Mx)
        My_k = np.fft.fftn(My)
        Mz_k = np.fft.fftn(Mz)
        Mx_k = np.fft.fftshift(Mx_k)
        My_k = np.fft.fftshift(My_k)
        Mz_k = np.fft.fftshift(Mz_k)

        Kz2 = K[:, :, :, 2]
        Cx = (1.0 / 6.0) * (1.0 - 3.0 * Kz2) * mu0
        Cy = (1.0 / 6.0) * (1.0 - 3.0 * Kz2) * mu0
        Cz = (2.0 / 6.0) * (3.0 * Kz2 - 1.0) * mu0

        Mx_k = Cx * Mx_k
        My_k = Cy * My_k
        Mz_k = Cz * Mz_k

        Mx_k = np.fft.ifftshift(Mx_k)
        My_k = np.fft.ifftshift(My_k)
        Mz_k = np.fft.ifftshift(Mz_k)

        Mx_d = np.fft.ifftn(Mx_k).real
        My_d = np.fft.ifftn(My_k).real
        Mz_d = np.fft.ifftn(Mz_k).real

        Mx_d = Mx_d[0:Nx, 0:Ny, 0:Nz]
        My_d = My_d[0:Nx, 0:Ny, 0:Nz]
        Mz_d = Mz_d[0:Nx, 0:Ny, 0:Nz]

        Bx_flat = Mx_d.reshape(Nspin)
        By_flat = My_d.reshape(Nspin)
        Bz_flat = Mz_d.reshape(Nspin)

        Wdx = -Gamma * Bx_flat * mask
        Wdy = -Gamma * By_flat * mask
        Wdz = -Gamma * Bz_flat * mask

        Mx_avg_dp = np.average(Mx_flat)
        My_avg_dp = np.average(My_flat)
        Mz_avg_dp = np.average(Mz_flat)

        coef = float(self.Demag_Coefficient)
        if abs(coef) > 0.0:
            factor = -Gamma * mu0 * coef
            Wdx = Wdx + factor * (-Mx_avg_dp)
            Wdy = Wdy + factor * (-My_avg_dp)
            Wdz = Wdz + factor * (2.0 * Mz_avg_dp)

        return Wdx, Wdy, Wdz

    def MeanDipolarFieldScipy(self, Mx_flat, My_flat, Mz_flat):
        """
        Mean dipole field (independent of FFT dipole), SciPy version.
        For now only z-component: Wz_mean = Mean_Dipolar_Strength * <Mz>.
        """
        if (not self.Mean_Dipolar_On) or (self.Mean_Dipolar_Strength == 0.0):
            Nspin = self.ChemicalShifts * self.Isochromats
            zeros = np.zeros(Nspin, dtype=self.DTYPE)
            return zeros, zeros, zeros
        Nspin = self.ChemicalShifts * self.Isochromats
        Mz_avg_mean = np.average(Mz_flat)
        Wdx_mean = np.zeros(Nspin, dtype=self.DTYPE)
        Wdy_mean = np.zeros(Nspin, dtype=self.DTYPE)
        Wdz_mean = self.Mean_Dipolar_Strength * Mz_avg_mean * np.ones(Nspin, dtype=self.DTYPE)
        return Wdx_mean, Wdy_mean, Wdz_mean

    def SelectJaxDevice(self):
        devices = jax.devices()
        if self.JAX_Device == "gpu":
            for d in devices:
                if d.platform == "gpu":
                    return d
            print("Warning: GPU requested but not found. Falling back to CPU.")
            return jax.devices("cpu")[0]
        else:
            return jax.devices("cpu")[0]

    def Evolution(self):
        self.Update()

        backend = str(self.ODE_Backend).lower()
        if backend == "scipy":
            self.EvolutionScipy()
        elif backend == "jax":
            if not JAX_AVAILABLE:
                raise RuntimeError("JAX / Diffrax backend selected but not available.")
            self.EvolutionJax()
        else:
            raise ValueError("Unknown ODE_Backend '" + str(self.ODE_Backend) + "'")

    def EvolutionScipy(self):
        M0 = self.M
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
        Mo_flat = self.Mo

        def MDOT(t, Mvec):
            Mx_local = Mvec[0::3]
            My_local = Mvec[1::3]
            Mz_local = Mvec[2::3]
            omega_RD = 1j * RD_Xi * (np.mean(Mx_local) + 1j * np.mean(My_local)) * np.exp(-1j * RD_Phase)
            B1_Field = B1_Amplitude * np.exp(1j * (B1_Frequency * t + B1_Phase))
            Wdx_dp, Wdy_dp, Wdz_dp = self.DipolarFieldScipy(Mvec)
            Wdx_mean, Wdy_mean, Wdz_mean = self.MeanDipolarFieldScipy(Mx_local, My_local, Mz_local)
            Wx = Omega_X + omega_RD.real + B1_Field.real + Wdx_dp + Wdx_mean
            Wy = Omega_Y + omega_RD.imag + B1_Field.imag + Wdy_dp + Wdy_mean
            Wz = Omega_Z + Wdz_dp + Wdz_mean
            Mdot = np.zeros_like(Mvec)
            Mdot[0::3] = -R2 * Mx_local - Wz * My_local - Wy * Mz_local
            Mdot[1::3] = Wz * Mx_local - R2 * My_local + Wx * Mz_local
            Mdot[2::3] = Wy * Mx_local - Wx * My_local - R1 * Mz_local + R1 * Mo_flat
            return Mdot

        start_time = time.time()
        Msol = solve_ivp(MDOT, [0.0, self.AQTime], M0, method=self.ODEMethod, t_eval=self.tpoints, atol=1.0e-10, rtol=1.0e-10)
        end_time = time.time()
        timetaken = end_time - start_time
        print("[SciPy] Total time = " + format(timetaken, ".6f") + " s")
        self.PostprocessSolution(Msol.t, Msol.y)
        print("Simulation is completed (SciPy backend).")

    def EvolutionJax(self):
        device = self.SelectJaxDevice()
        y0 = jax.device_put(self.M, device=device)
        Mo_flat = jax.device_put(self.Mo, device=device)
        Omega_Z = jax.device_put(self.Omega_Z, device=device)

        Mask_flat = jax.device_put(self.Mask, device=device)
        Dipolar_On_flag = 1 if self.Dipolar_On else 0
        Demag_Coefficient = float(self.Demag_Coefficient)

        # Safe KTensor handling (works for mean dipole or FFT dipole)
        if self.Dipolar_On:
            if self.KTensor is None:
                self.BuildKSpaceLattice()
            KTensor_z2_np = self.KTensor[:, :, :, 2]
        else:
            # dummy array, never used when Dipolar_On_flag == 0
            KTensor_z2_np = np.zeros((1, 1, 1), dtype=self.DTYPE)

        KTensor_z2 = jax.device_put(KTensor_z2_np, device=device)

        Mean_Dipolar_On_flag = 1 if self.Mean_Dipolar_On else 0
        Mean_Dipolar_Strength_val = float(self.Mean_Dipolar_Strength)

        args = (self.Isochromats, self.ChemicalShifts,
                self.RD_Xi, self.RD_Phase,
                self.Omega_X, self.Omega_Y, Omega_Z,
                self.Relaxation_R1, self.Relaxation_R2,
                self.B1_Amplitude, self.B1_Frequency, self.B1_Phase,
                Mo_flat,
                Dipolar_On_flag, Demag_Coefficient,
                Mean_Dipolar_On_flag, Mean_Dipolar_Strength_val,
                int(self.Lattice_Nx), int(self.Lattice_Ny), int(self.Lattice_Nz),
                int(self.Padd_X), int(self.Padd_Y), int(self.Padd_Z),
                self.Gamma, self.Permeability,
                Mask_flat, KTensor_z2)

        term = dfx.ODETerm(MDOT_Jax)
        solver = dfx.BDF2() if self.ODE_Stiff else dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=1.0e-10, atol=1.0e-10)
        saveat = dfx.SaveAt(ts=jnp.array(self.tpoints))

        start_time = time.time()
        sol = dfx.diffeqsolve(term, solver,
                              t0=0.0, t1=float(self.AQTime),
                              dt0=float(self.DT),
                              y0=y0, args=args,
                              saveat=saveat,
                              stepsize_controller=stepsize_controller,
                              max_steps=self.JAX_MaxSteps)
        end_time = time.time()
        timetaken = end_time - start_time
        print("[JAX+Diffrax] Total time = " + format(timetaken, ".6f") + " s")

        ts = np.asarray(sol.ts, dtype=float)
        ys = np.asarray(sol.ys, dtype=float)
        ys = ys.T
        self.PostprocessSolution(ts, ys)
        print("Simulation is completed (JAX backend).")

    def PostprocessSolution(self, t_array, M_array):
        """
        Post-process the ODE output.
        Prints all key variable names and shapes,
        and stores the final M for next-run initialization.
        """

        # -----------------------------
        # Store solution
        # -----------------------------
        self.tpoints = t_array                    # shape (Nt,)
        self.Mpoints = M_array                    # shape (3*Nspin, Nt)

        # Compute summed magnetization components
        self.Mx = np.sum(self.Mpoints[0::3, :], axis=0)   # (Nt,)
        self.My = np.sum(self.Mpoints[1::3, :], axis=0)
        self.Mz = np.sum(self.Mpoints[2::3, :], axis=0)
        self.Mabs = np.sqrt(self.Mx**2 + self.My**2)

        # Complex transverse signal
        self.Signal = self.Mx + 1j * self.My

        # Derived sampling params
        self.DT = self.tpoints[1] - self.tpoints[0]
        self.FS = 1.0 / self.DT

        # FFT for spectrum
        Spectrum = np.fft.fft(self.Signal)
        self.Spectrum = np.fft.fftshift(Spectrum)
        self.Freq = np.linspace(-self.FS / 2.0,
                                self.FS / 2.0,
                                self.Signal.shape[-1])

        # -----------------------------
        # STORE FINAL M FOR NEXT SIMULATION
        # -----------------------------
        # Last column of M_array is the final magnetization state.
        self.M = self.Mpoints[:, -1].copy()

        # -----------------------------
        # PRINT VARIABLE SHAPES
        # -----------------------------
        print("\n========== POST-PROCESS VARIABLES ==========\n")

        def p(name, arr):
            try:
                print(f"{name:20s} : shape = {np.shape(arr)}")
            except:
                print(f"{name:20s} : (not array)")

        # Time domain quantities
        p("tpoints", self.tpoints)
        p("Mpoints", self.Mpoints)
        p("Mx", self.Mx)
        p("My", self.My)
        p("Mz", self.Mz)
        p("Mabs", self.Mabs)
        p("Signal", self.Signal)

        # Spectrum
        p("Spectrum", self.Spectrum)
        p("Freq", self.Freq)

        # Lattice and dipole fields
        p("Omega_Z", self.Omega_Z)
        p("Mask", self.Mask)
        if self.KTensor is not None:
            p("KTensor", self.KTensor)

        # Final magnetization
        p("M", self.M)

        print("\n============================================\n")

        # -----------------------------
        # APPEND TIME + SIGNAL SEQUENTIALLY
        # -----------------------------
        if self.tpointsFull is None:
            # First acquisition → no time offset
            self.tpointsFull = self.tpoints.copy()
            self.SignalFull = self.Signal.copy()
        else:
            # Time offset = last time in the previous block + DT
            offset = self.tpointsFull[-1] + self.DT
            t_shifted = self.tpoints + offset

            # Append
            self.tpointsFull = np.concatenate((self.tpointsFull, t_shifted))
            self.SignalFull = np.concatenate((self.SignalFull, self.Signal))

    def Ploting_Signal(self):
        rc("font", weight="bold")
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=(15, 5))
        spec = fig.add_gridspec(1, 1)
        self.fig_counter = self.fig_counter + 1
        ax1 = fig.add_subplot(spec[0, 0])
        ax1.plot(self.tpointsFull, self.SignalFull, linewidth=3.0, color="blue", label="Signal")
        ax1.set_xlabel("Time (s)", fontsize=25, color="black", fontweight="bold")
        ax1.set_ylabel("$M_{T}$ (AU)", fontsize=25, color="blue", fontweight="bold")
        ax1.legend(fontsize=25, frameon=False)
        ax1.tick_params(axis="both", labelsize=14)
        ax1.grid(True, linestyle="-.")
        ax1.set_xlim(self.Plot_Xlim)
        ax1.set_ylim(self.Plot_Ylim)
        if self.Plot_Save:
            plt.savefig("Signal.pdf", bbox_inches="tight")

    def Ploting_MxMyMz(self):
        rc("font", weight="bold")
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=(15, 5))
        spec = fig.add_gridspec(1, 1)
        self.fig_counter = self.fig_counter + 1
        ax1 = fig.add_subplot(spec[0, 0])
        ax1.plot(self.tpoints, self.Mx, linewidth=3.0, color="blue", label="Mx")
        ax1.plot(self.tpoints, self.My, linewidth=3.0, color="green", label="My")
        ax1.set_xlabel("Time (s)", fontsize=25, color="black", fontweight="bold")
        ax1.set_ylabel("$M_{T}$ (AU)", fontsize=25, color="blue", fontweight="bold")
        ax1.legend(fontsize=25, frameon=False)
        ax1.tick_params(axis="both", labelsize=14)
        ax1.grid(True, linestyle="-.")
        ax1.set_xlim(self.Plot_Xlim)
        ax1.set_ylim(self.Plot_Ylim)
        ax10 = ax1.twinx()
        ax10.plot(self.tpoints, self.Mz, linewidth=3.0, color="red", label="Mz")
        ax10.set_xlabel("Time (s)", fontsize=30, color="black", fontweight="bold")
        ax10.set_ylabel("$M_{Z}$ (AU)", fontsize=30, color="red", fontweight="bold")
        ax10.legend(fontsize=30, frameon=False)
        ax10.tick_params(axis="both", labelsize=20)
        ax1.set_xlim(self.Plot_Xlim)
        ax1.set_ylim(self.Plot_Ylim)
        if self.Plot_Save:
            plt.savefig("MxMyMz.pdf", bbox_inches="tight")

    def Ploting_Spectrum(self):
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=(15, 5))
        spec = fig.add_gridspec(1, 1)
        self.fig_counter = self.fig_counter + 1
        ax1 = fig.add_subplot(spec[0, 0])
        ax1.plot(self.Freq, self.Spectrum, linewidth=3.0, color="black")
        ax1.set_xlabel("Frequency (Hz)", fontsize=25, color="green", fontweight="bold")
        ax1.set_ylabel("Spectrum (AU)", fontsize=25, color="black", fontweight="bold")
        ax1.tick_params(axis="both", labelsize=14)
        ax1.grid(True, linestyle="-.")
        ax1.set_xlim(self.Plot_Xlim)
        ax1.set_ylim(self.Plot_Ylim)
        if self.Plot_Save:
            plt.savefig("Spectrum.pdf", bbox_inches="tight")

    def Plotting_Sphere(self):
        S_phi = np.linspace(0.0, np.pi, 20)
        S_theta = np.linspace(0.0, 2.0 * np.pi, 20)
        S_phi, S_theta = np.meshgrid(S_phi, S_theta)
        S_x = np.sum(self.Magnetization) * np.sin(S_phi) * np.cos(S_theta)
        S_y = np.sum(self.Magnetization) * np.sin(S_phi) * np.sin(S_theta)
        S_z = np.sum(self.Magnetization) * np.cos(S_phi)
        tlim1 = 0
        tlim2 = -1
        ax = plt.figure(self.fig_counter, figsize=(10, 10)).add_subplot(projection="3d")
        self.fig_counter = self.fig_counter + 1
        ax.plot_wireframe(S_x, S_y, S_z, color="cyan", linewidth=1.0)
        ax.plot(self.Mx[tlim1:tlim2], self.My[tlim1:tlim2], self.Mz[tlim1:tlim2], color="black", linewidth=1.0)
        ax.view_init(10, 20)
        ax.set_xlabel("My", fontsize=14, color="black", fontweight="bold")
        ax.set_ylabel("Mx", fontsize=14, color="black", fontweight="bold")
        ax.set_zlabel("Mz", fontsize=14, color="black", fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="-.")
        if self.Plot_Save:
            plt.savefig("Sphere.pdf", bbox_inches="tight")
        plt.show()

    def Plotting_Lattice(self, show_all_points=True):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        Nx = int(self.Lattice_Nx)
        Ny = int(self.Lattice_Ny)
        Nz = int(self.Lattice_Nz)

        if self.Mask3D is None:
            self.BuildShapeMask()

        mask = self.Mask3D

        x = np.arange(Nx, dtype=self.DTYPE) - 0.5 * (Nx - 1)
        y = np.arange(Ny, dtype=self.DTYPE) - 0.5 * (Ny - 1)
        z = np.arange(Nz, dtype=self.DTYPE) - 0.5 * (Nz - 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        Xf = X.ravel()
        Yf = Y.ravel()
        Zf = Z.ravel()
        Mf = mask.ravel()

        X_in = Xf[Mf > 0.5]
        Y_in = Yf[Mf > 0.5]
        Z_in = Zf[Mf > 0.5]

        fig = plt.figure(self.fig_counter, figsize=(8, 8))
        self.fig_counter = self.fig_counter + 1
        ax = fig.add_subplot(111, projection='3d')

        if show_all_points:
            ax.scatter(Xf, Yf, Zf, s=5, c='0.85', marker='o', alpha=0.4, label='Lattice')

        ax.scatter(X_in, Y_in, Z_in, s=15, c='red', marker='o', alpha=0.9, label='Sample')

        ax.set_xlabel('x (lattice units)', fontsize=12, fontweight='bold')
        ax.set_ylabel('y (lattice units)', fontsize=12, fontweight='bold')
        ax.set_zlabel('z (lattice units)', fontsize=12, fontweight='bold')

        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_ylim(y[0] - 0.5, y[-1] + 0.5)
        ax.set_zlim(z[0] - 0.5, z[-1] + 0.5)

        shape_name = str(self.Lattice_Shape).capitalize()
        ax.set_title('Lattice (' + shape_name + ')', fontsize=14, fontweight='bold')
        ax.view_init(elev=20, azim=40)
        ax.legend(frameon=False)

        ax.grid(True, linestyle='-.')
        plt.tight_layout()
        plt.show()

    def Plotting_FourierAnalyzer(self):
        self.SetupPlot()
        self.ConnectEvents()

    def SetupPlot(self):
        self.figsize = (12, 9)
        self.fig = plt.figure(self.fig_counter, figsize=self.figsize)
        self.ax = self.fig.subplots(2, 2)
        self.fig_counter = self.fig_counter + 1
        (self.line1,) = self.ax[0, 0].plot(self.tpoints, self.Signal.real, '-', color='green')
        self.ax[0, 0].set_title("Time Domain")
        self.ax[0, 0].set_xlabel("Time [s]")
        self.ax[0, 0].set_ylabel("Signal")
        self.ax[0, 0].grid()
        self.vline1 = self.ax[0, 1].axvline(color='k', lw=0.8, ls='--')
        self.vline2 = self.ax[0, 1].axvline(color='k', lw=0.8, ls='--')
        self.text1 = self.ax[0, 1].text(0.0, 0.95, '', transform=self.ax[0, 1].transAxes)
        spectrum_data = np.abs(self.Spectrum) if self.abs_spectrum else self.Spectrum
        (self.line2,) = self.ax[0, 1].plot(self.Freq, spectrum_data, '-', color='green')
        self.ax[0, 1].set_title("Frequency Domain (Top)")
        self.ax[0, 1].set_xlabel("Frequency [Hz]")
        self.ax[0, 1].set_ylabel("Spectrum")
        if self.Plot_Xlim is not None:
            self.ax[0, 1].set_xlim(self.Plot_Xlim)
        self.ax[0, 1].grid()
        (self.line3,) = self.ax[1, 0].plot(self.Freq, spectrum_data, '-', color='green')
        self.ax[1, 0].set_title("Frequency Domain (Bottom)")
        self.ax[1, 0].set_xlabel("Frequency [Hz]")
        self.ax[1, 0].set_ylabel("Spectrum")
        if self.Plot_Xlim is not None:
            self.ax[1, 0].set_xlim(self.Plot_Xlim)
        self.ax[1, 0].grid()
        self.vline3 = self.ax[1, 1].axvline(color='k', lw=0.8, ls='--')
        self.vline4 = self.ax[1, 1].axvline(color='k', lw=0.8, ls='--')
        self.text2 = self.ax[1, 1].text(0.0, 0.95, '', transform=self.ax[1, 1].transAxes)
        (self.line4,) = self.ax[1, 1].plot(self.tpoints, self.Signal.real, '-', color='green')
        self.ax[1, 1].set_title("Reconstructed Signal")
        self.ax[1, 1].set_xlabel("Time [s]")
        self.ax[1, 1].set_ylabel("Signal")
        self.ax[1, 1].grid()

    def ConnectEvents(self):
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
    """

    def __init__(self, Mx, My, spectrum, ax, fig, line1, line2, line3, line4,
                 vline1, vline2, vline3, vline4, text1, text2, Abs_Sp):
        (self.x1, self.y1) = line1.get_data()
        (self.x2, self.y2) = line2.get_data()
        (self.x3, self.y3) = line3.get_data()
        (self.x4, self.y4) = line4.get_data()
        self.dt = self.x1[1] - self.x1[0]
        self.fs = 1.0 / self.dt
        self.ax = ax
        self.fig = fig
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
        self.x1in = None
        self.x1fi = None
        self.x2in = None
        self.x2fi = None
        self.x3in = None
        self.x3fi = None
        self.x4in = None
        self.x4fi = None

    def button_press(self, event):
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
        if event.inaxes is self.ax[0, 0]:
            self.x1fi = min(np.searchsorted(self.x1, event.xdata), len(self.x1) - 1)
            self.ax[0, 0].axvspan(self.x1[self.x1in], self.x1[self.x1fi], color='red', alpha=0.2)
            Spectrum = np.fft.fft(self.Mt[self.x1in:self.x1fi])
            Spectrum = np.fft.fftshift(Spectrum)
            spectrum = Spectrum
            freq = np.linspace(-self.fs / 2, self.fs / 2, spectrum.shape[-1])
            for line in self.ax[0, 1].lines[1:]:
                line.remove()
            self.ax[0, 1].plot(freq, np.abs(spectrum) if self.Abs_Sp else spectrum, '-', color='red')
            plt.draw()
        elif event.inaxes is self.ax[1, 0]:
            self.x3fi = min(np.searchsorted(self.x3, event.xdata), len(self.x3) - 1)
            window = np.zeros_like(self.y3)
            window[self.x3in:self.x3fi] = 1.0
            self.ax[1, 0].axvspan(self.x3[self.x3in], self.x3[self.x3fi], color='red', alpha=0.2)
            Sig = np.fft.ifftshift(self.spectrum * window)
            Sig = np.fft.ifft(Sig)
            sig = Sig
            t = np.linspace(0, self.dt * len(self.y3), len(self.y3))
            for line in self.ax[1, 1].lines[1:]:
                line.remove()
            self.ax[1, 1].plot(self.x4, self.y4, '-', color='blue')
            self.ax[1, 1].plot(t, sig.real, '-', color='red')
            plt.draw()
        elif event.inaxes is self.ax[0, 1]:
            self.x2fi = event.xdata
            self.vline2.set_xdata([self.x2fi])
            self.text1.set_text('Freq = ' + format(abs(self.x2fi - self.x2in), ".5f") + ' Hz')
            plt.draw()
        elif event.inaxes is self.ax[1, 1]:
            self.x4fi = event.xdata
            self.vline4.set_xdata([self.x4fi])
            self.text2.set_text('Time = ' + format(abs(self.x4fi - self.x4in), ".5f") + ' s')
            plt.draw()
