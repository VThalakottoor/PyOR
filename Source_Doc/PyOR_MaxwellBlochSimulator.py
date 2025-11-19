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

        # Mean Field Dipolar Field
        self.Mean_Dipolar_On = False
        self.Mean_Dipolar_Strength = 0.0

        # Secular Field Dipolar Field
        self.Secular_Dipolar_On = False
        self.Secular_Dipolar_Strength = 0.0
        self.Secular_Dipolar_Axis = np.array([0.0, 0.0, 1.0], dtype=self.DTYPE)

        # Pair wise Dipolar interaction
        self.Positions = None  # shape (N, 3)
        self.Pair_Dipolar_On = False
        self.Pair_Dipolar_Strength = 0.0

        # FFT-based (continuum) dipolar field
        self.FFT_Dipolar_On = False
        self.FFT_Dipolar_Strength = 0.0   # absorbs mu0, gamma^2, etc.
        self.FFT_Padding = (0, 0, 0)      # (paddx, paddy, paddz)
        self.LatticeShape = None          # (nx, ny, nz) for 3D lattice
        self.FFT_Dxx = None
        self.FFT_Dyy = None
        self.FFT_Dzz = None

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

    def Mean_DipolarField(self,Mx,My,Mz):
        """
        Simple mean-field dipolar term:
        returns (Wdx, Wdy, Wdz) in rad/s for each spin.

        For now:
            Wdz = D * <Mz>
            Wdx = Wdy = 0
        where <Mz> is the average over all spins.
        """
        if (not self.Mean_Dipolar_On) or (self.Mean_Dipolar_Strength == 0.0):
            return 0.0, 0.0, 0.0

        # Global average magnetization (bulk)
        Mz_avg = np.mean(Mz)

        # Effective dipolar frequency along z for *all* spins
        Wdz = self.Mean_Dipolar_Strength * Mz_avg

        # If you later want transverse parts, you can add them here:
        Wdx = 0.0
        Wdy = 0.0

        return Wdx, Wdy, Wdz     

    def Secular_dipolar_field(self, Mx, My, Mz):
        """
        Secular dipolar mean-field term.

        W_dip = D * T * <M>

        Returns scalar (Wdx, Wdy, Wdz) in rad/s to be added
        to the effective precession frequencies for all spins.
        """
        if (not self.Secular_Dipolar_On) or (self.Secular_Dipolar_Strength == 0.0):
            return 0.0, 0.0, 0.0

        # Bulk magnetization
        M_avg = np.array([np.mean(Mx), np.mean(My), np.mean(Mz)], dtype=self.DTYPE)

        # Ensure axis is unit length
        n = np.array(self.Secular_Dipolar_Axis, dtype=self.DTYPE)
        n_norm = np.linalg.norm(n)
        if n_norm == 0.0:
            # fallback: no dipolar field if axis is invalid
            return 0.0, 0.0, 0.0
        n /= n_norm

        # Secular dipolar tensor T_ij = 3 n_i n_j - delta_ij
        T = 3.0 * np.outer(n, n) - np.eye(3, dtype=self.DTYPE)

        # Dipolar frequency vector in rad/s
        W_dip = self.Secular_Dipolar_Strength * (T @ M_avg)

        return W_dip[0], W_dip[1], W_dip[2]

    def Pairwise_DipolarField(self, Mx, My, Mz):
        """
        Compute classical pairwise dipolar field at each spin (no FFT).

        Uses:
            B_i = C * sum_{j!=i} [ (3 (m_j·ê_ij) ê_ij - m_j) / r_ij^3 ]

        Returns:
            Wx, Wy, Wz  (arrays of length N) in rad/s
        """
        if (not getattr(self, "Pair_Dipolar_On", False)) or (self.Pair_Dipolar_Strength == 0.0):
            # Return zeros matching the shape of Mx/My/Mz
            zeros = np.zeros_like(Mx, dtype=self.DTYPE)
            return zeros, zeros, zeros

        if self.Positions is None:
            raise ValueError("Pairwise_DipolarField: self.Positions is not set.")

        # Flatten positions to (N, 3)
        r = np.asarray(self.Positions, dtype=self.DTYPE)
        if r.shape[0] != Mx.shape[0] or r.shape[1] != 3:
            raise ValueError("Positions must have shape (N, 3) with N = len(Mx).")

        N = Mx.shape[0]

        # Magnetic moment vectors from magnetization components
        M_vec = np.stack((Mx, My, Mz), axis=1)  # shape (N, 3)

        # Output arrays (local dipolar angular frequencies at each spin)
        Wx = np.zeros(N, dtype=self.DTYPE)
        Wy = np.zeros(N, dtype=self.DTYPE)
        Wz = np.zeros(N, dtype=self.DTYPE)

        C = self.Pair_Dipolar_Strength  # absorbs mu0, gamma^2, etc.

        for i in range(N):
            # Vector from all j to i
            dr = r[i] - r         # shape (N, 3)
            # Avoid self-interaction
            dr[i] = 0.0

            # Squared distances
            r2 = np.einsum("ij,ij->i", dr, dr)  # shape (N,)

            # Mask out i and any zero-distance pairs
            mask = r2 > 0.0
            if not np.any(mask):
                continue

            dr_valid = dr[mask]              # (N_valid, 3)
            Mj_valid = M_vec[mask]           # (N_valid, 3)
            r2_valid = r2[mask]              # (N_valid,)

            r_valid = np.sqrt(r2_valid)      # |r_ij|
            inv_r3 = 1.0 / (r2_valid * r_valid)  # 1/r^3

            # Unit vectors ê_ij
            e = dr_valid / r_valid[:, None]  # (N_valid, 3)

            # m_j · ê_ij
            mdot_e = np.einsum("ij,ij->i", Mj_valid, e)   # (N_valid,)

            # 3(m_j·ê) ê - m_j
            term = 3.0 * mdot_e[:, None] * e - Mj_valid   # (N_valid, 3)

            # Sum over j: Σ_j ( term_j / r_ij^3 )
            Bi = C * np.sum(term * inv_r3[:, None], axis=0)  # (3,)

            # Convert to angular freq (if C already includes -gamma, then this is W directly)
            Wx[i], Wy[i], Wz[i] = Bi[0], Bi[1], Bi[2]

        return Wx, Wy, Wz

    def FFT_DipolarField(self, Mx, My, Mz):
        """
        Compute dipolar field using FFT-based convolution on a regular 3D lattice.

        Assumes:
        - self.LatticeShape = (nx, ny, nz) and nx*ny*nz == len(Mx)
        - self.FFT_Dxx, FFT_Dyy, FFT_Dzz have been built
          by Build_Dipolar_Kernel_FFT(padding=...)
        - If self.GeometryMask or self.CylinderMask is present, magnetization
          outside the mask is set to zero before convolution.

        Parameters
        ----------
        Mx, My, Mz : 1D arrays of length N
            Magnetization components at each lattice site.

        Returns
        -------
        Wx, Wy, Wz : 1D arrays of length N
            Dipolar contributions to angular frequency (rad/s) at each site.
        """
        if (not self.FFT_Dipolar_On) or (self.FFT_Dipolar_Strength == 0.0):
            zeros = np.zeros_like(Mx, dtype=self.DTYPE)
            return zeros, zeros, zeros

        if self.LatticeShape is None:
            raise ValueError(
                "FFT_DipolarField: self.LatticeShape is None. "
                "Call Build_3D_Lattice_Positions or a lattice builder first."
            )
        if self.FFT_Dxx is None or self.FFT_Dyy is None or self.FFT_Dzz is None:
            raise ValueError(
                "FFT_DipolarField: FFT kernels not built. "
                "Call Build_Dipolar_Kernel_FFT(padding=...) first."
            )

        nx, ny, nz = self.LatticeShape
        N_expected = nx * ny * nz
        if Mx.shape[0] != N_expected:
            raise ValueError(
                f"FFT_DipolarField: len(Mx)={Mx.shape[0]} but nx*ny*nz={N_expected}."
            )

        paddx, paddy, paddz = self.FFT_Padding
        Nx = nx + paddx
        Ny = ny + paddy
        Nz = nz + paddz

        # reshape to 3D lattice
        Mx3 = Mx.reshape((nx, ny, nz), order='C')
        My3 = My.reshape((nx, ny, nz), order='C')
        Mz3 = Mz.reshape((nx, ny, nz), order='C')

        # ------------------------------------------------------------------
        # Apply geometry masks: GeometryMask (sphere) takes priority,
        # otherwise CylinderMask if available. Outside mask -> magnetization = 0.
        # ------------------------------------------------------------------
        mask3 = None

        # Sphere/geometry mask (3D)
        geom_mask = getattr(self, "GeometryMask", None)
        if geom_mask is not None:
            mask3 = np.asarray(geom_mask, dtype=bool)
            if mask3.shape != (nx, ny, nz):
                raise ValueError(
                    f"GeometryMask shape {mask3.shape} inconsistent with "
                    f"LatticeShape {(nx, ny, nz)}."
                )
        else:
            # Cylinder mask (1D flattened)
            cyl_mask = getattr(self, "CylinderMask", None)
            if cyl_mask is not None:
                cyl_mask = np.asarray(cyl_mask, dtype=bool)
                if cyl_mask.shape != (N_expected,):
                    raise ValueError(
                        f"CylinderMask shape {cyl_mask.shape} inconsistent with "
                        f"N = {N_expected}."
                    )
                mask3 = cyl_mask.reshape((nx, ny, nz), order='C')

        if mask3 is not None:
            Mx3 = np.where(mask3, Mx3, 0.0)
            My3 = np.where(mask3, My3, 0.0)
            Mz3 = np.where(mask3, Mz3, 0.0)

        # zero padding on high side
        Mx_pad = np.pad(Mx3, ((0, paddx), (0, paddy), (0, paddz)), mode='constant')
        My_pad = np.pad(My3, ((0, paddx), (0, paddy), (0, paddz)), mode='constant')
        Mz_pad = np.pad(Mz3, ((0, paddx), (0, paddy), (0, paddz)), mode='constant')

        # FFT
        Mx_k = np.fft.fftn(Mx_pad)
        My_k = np.fft.fftn(My_pad)
        Mz_k = np.fft.fftn(Mz_pad)

        # Shift so that k=0 is in the center (to match constructed kernels)
        Mx_k = np.fft.fftshift(Mx_k)
        My_k = np.fft.fftshift(My_k)
        Mz_k = np.fft.fftshift(Mz_k)

        Dxx = self.FFT_Dxx
        Dyy = self.FFT_Dyy
        Dzz = self.FFT_Dzz

        if Dxx.shape != (Nx, Ny, Nz):
            raise ValueError(
                f"FFT_Dxx shape {Dxx.shape} inconsistent with padded grid {(Nx, Ny, Nz)}."
            )

        c = self.FFT_Dipolar_Strength  # overall scaling (physics constants)

        # Apply kernel component-wise in k-space
        Wx_k = c * Dxx * Mx_k
        Wy_k = c * Dyy * My_k
        Wz_k = c * Dzz * Mz_k

        # Back to real space
        Wx_k = np.fft.ifftshift(Wx_k)
        Wy_k = np.fft.ifftshift(Wy_k)
        Wz_k = np.fft.ifftshift(Wz_k)

        Wx_pad = np.fft.ifftn(Wx_k).real
        Wy_pad = np.fft.ifftn(Wy_k).real
        Wz_pad = np.fft.ifftn(Wz_k).real

        # crop back to original lattice
        Wx3 = Wx_pad[0:nx, 0:ny, 0:nz]
        Wy3 = Wy_pad[0:nx, 0:ny, 0:nz]
        Wz3 = Wz_pad[0:nx, 0:ny, 0:nz]

        # flatten to 1D
        Wx = Wx3.ravel(order='C')
        Wy = Wy3.ravel(order='C')
        Wz = Wz3.ravel(order='C')

        return Wx, Wy, Wz


    def Build_Dipolar_Kernel_FFT(self, padding=(0, 0, 0)):
        """
        Build the k-space dipolar kernel Dxx, Dyy, Dzz for FFT-based dipolar field.

        Works on a regular 3D lattice defined by self.LatticeShape = (nx, ny, nz).
        The kernel is the usual demagnetization tensor in k-space:
            D_ij(k) ∝ (δ_ij / 3 - k_i k_j / k^2)
        Here we keep only the diagonal elements (xx, yy, zz).

        Parameters
        ----------
        padding : tuple of 3 ints
            Extra zeros along x, y, z (on the high side) to reduce wrap-around
            artefacts in the FFT convolution.

        Sets
        ----
        self.FFT_Padding : (paddx, paddy, paddz)
        self.FFT_Dxx, FFT_Dyy, FFT_Dzz : ndarrays of shape (Nx, Ny, Nz)
        """
        if self.LatticeShape is None:
            raise ValueError(
                "Build_Dipolar_Kernel_FFT: self.LatticeShape is None. "
                "Call Build_3D_Lattice_Positions first."
            )

        nx, ny, nz = self.LatticeShape
        paddx, paddy, paddz = padding

        Nx = nx + paddx
        Ny = ny + paddy
        Nz = nz + paddz

        self.FFT_Padding = (paddx, paddy, paddz)

        # Dimensionless k-grid from -0.5 to +0.5 (like in MASER)
        kx = np.linspace(-0.5, 0.5, Nx, endpoint=True, dtype=self.DTYPE)
        ky = np.linspace(-0.5, 0.5, Ny, endpoint=True, dtype=self.DTYPE)
        kz = np.linspace(-0.5, 0.5, Nz, endpoint=True, dtype=self.DTYPE)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2

        # Avoid division by zero at k = 0
        K2_safe = np.where(K2 == 0.0, 1.0, K2)

        # Unit-vector components k_i / |k|
        KX_hat = np.where(K2 > 0.0, KX / np.sqrt(K2_safe), 0.0)
        KY_hat = np.where(K2 > 0.0, KY / np.sqrt(K2_safe), 0.0)
        KZ_hat = np.where(K2 > 0.0, KZ / np.sqrt(K2_safe), 0.0)

        # Demagnetization tensor diagonal: D_ii ∝ 1/3 - k_i^2 / k^2
        Dxx = (1.0 / 3.0) - (KX_hat**2)
        Dyy = (1.0 / 3.0) - (KY_hat**2)
        Dzz = (1.0 / 3.0) - (KZ_hat**2)

        # At k = 0, define kernel to be 0 (no global offset field)
        Dxx[K2 == 0.0] = 0.0
        Dyy[K2 == 0.0] = 0.0
        Dzz[K2 == 0.0] = 0.0

        self.FFT_Dxx = Dxx
        self.FFT_Dyy = Dyy
        self.FFT_Dzz = Dzz

    def Build_Line_Positions(self, spacing=1.0, axis='z'):
        """
        Build positions for spins on a 1D line, centered at 0.

        Parameters
        ----------
        spacing : float
            Distance between neighboring spins (arbitrary units or meters).
        axis : str
            Which axis to put the line along: 'x', 'y', or 'z'.

        Sets
        ----
        self.Positions : ndarray, shape (N, 3)
        """
        N = self.ChemicalShifts * self.Isochromats

        # Coordinates along the chosen axis, centered at zero
        idx = np.arange(N, dtype=self.DTYPE)
        center = 0.5 * (N - 1)
        coord = (idx - center) * spacing  # shape (N,)

        # Initialize all positions to zero
        pos = np.zeros((N, 3), dtype=self.DTYPE)

        if axis.lower() == 'x':
            pos[:, 0] = coord
        elif axis.lower() == 'y':
            pos[:, 1] = coord
        elif axis.lower() == 'z':
            pos[:, 2] = coord
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        self.Positions = pos

    def Build_3D_Lattice_Positions(self, nx, ny, nz, spacing=1.0):
        """
        Build positions for spins on a 3D rectangular lattice, centered at 0.

        Parameters
        ----------
        nx, ny, nz : int
            Number of spins along x, y, z.
            Must satisfy nx * ny * nz == ChemicalShifts * Isochromats.
        spacing : float
            Lattice spacing (same in x, y, z).

        Sets
        ----
        self.Positions : ndarray, shape (N, 3)
        """
        N = self.ChemicalShifts * self.Isochromats
        if nx * ny * nz != N:
            raise ValueError(
                f"nx*ny*nz = {nx*ny*nz} must equal N = {N} "
                "(ChemicalShifts * Isochromats)."
            )
        
        self.LatticeShape = (nx, ny, nz)

        # Index ranges, centered at zero
        x_idx = np.arange(nx, dtype=self.DTYPE)
        y_idx = np.arange(ny, dtype=self.DTYPE)
        z_idx = np.arange(nz, dtype=self.DTYPE)

        x_center = 0.5 * (nx - 1)
        y_center = 0.5 * (ny - 1)
        z_center = 0.5 * (nz - 1)

        x = (x_idx - x_center) * spacing
        y = (y_idx - y_center) * spacing
        z = (z_idx - z_center) * spacing

        # 3D meshgrid -> lattice
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # shape (nx, ny, nz)

        # Flatten to (N, 3), consistent with your flattening of M
        pos = np.column_stack((
            X.ravel(order='C'),
            Y.ravel(order='C'),
            Z.ravel(order='C')
        ))

        self.Positions = pos

    def Build_Sphere_Volume_Positions(self, radius=1.0, spacing=1.0):
        """
        Define a spherical sample on a regular 3D grid, suitable for FFT-based dipolar fields.

        - If self.LatticeShape is already defined (e.g. via Build_3D_Lattice_Positions),
        we reuse that grid and only build a spherical mask.

        - Otherwise, we build a cubic grid nx = ny = nz such that nx^3 = N,
        where N = ChemicalShifts * Isochromats. This only works if N is a perfect cube.

        Sets
        ----
        self.Positions   : (N, 3) array of grid coordinates
        self.LatticeShape: (nx, ny, nz)
        self.GeometryMask: (nx, ny, nz) array, 1 inside sphere, 0 outside
        """
        N = self.ChemicalShifts * self.Isochromats

        # If no lattice grid yet, try to build a cubic one
        if self.LatticeShape is None:
            n = int(round(N ** (1.0 / 3.0)))
            if n**3 != N:
                raise ValueError(
                    f"For FFT sphere with automatic grid, N = {N} must be a perfect cube. "
                    "Either choose N = nx*ny*nz with nx=ny=nz, or call "
                    "Build_3D_Lattice_Positions(nx, ny, nz, spacing) first."
                )
            # This will set self.LatticeShape and self.Positions
            self.Build_3D_Lattice_Positions(n, n, n, spacing=spacing)

        # Use the existing lattice
        nx, ny, nz = self.LatticeShape
        pos = np.asarray(self.Positions, dtype=self.DTYPE)
        if pos.shape != (nx*ny*nz, 3):
            raise ValueError(
                f"Positions shape {pos.shape} is inconsistent with LatticeShape {self.LatticeShape}."
            )

        # Reshape positions to (nx, ny, nz, 3)
        pos_4d = pos.reshape((nx, ny, nz, 3), order='C')
        x = pos_4d[..., 0]
        y = pos_4d[..., 1]
        z = pos_4d[..., 2]

        r2 = x**2 + y**2 + z**2

        # Spherical mask: 1 inside radius, 0 outside
        mask = (r2 <= radius**2).astype(self.DTYPE)

        # Store geometry mask
        self.GeometryMask = mask

    def Build_Cylinder_Lattice_Positions(self,
                                        nx, ny, nz,
                                        spacing=1.0,
                                        radius=None,
                                        height=None):
        """
        Build a 3D rectangular lattice (nx × ny × nz) suitable for FFT-based
        dipolar convolution, and mark which sites lie inside a cylinder
        aligned with the z-axis.

        The cylinder is defined by:
            x^2 + y^2 <= radius^2
            |z| <= height/2

        Parameters
        ----------
        nx, ny, nz : int
            Number of lattice points along x, y, z.
            Must satisfy nx * ny * nz == ChemicalShifts * Isochromats.
        spacing : float
            Lattice spacing (same in x, y, z).
        radius : float or None
            Cylinder radius. If None, it is chosen to fit inside the lattice:
                radius = 0.5 * min(nx, ny) * spacing
        height : float or None
            Cylinder height (along z). If None, use full lattice height:
                height = nz * spacing

        Sets
        ----
        self.Positions    : ndarray, shape (N, 3), regular lattice coordinates
        self.LatticeShape : (nx, ny, nz)
        self.CylinderMask : ndarray, shape (N,), boolean mask for cylinder sites
        """
        N = self.ChemicalShifts * self.Isochromats
        if nx * ny * nz != N:
            raise ValueError(
                f"nx*ny*nz = {nx*ny*nz} must equal N = {N} "
                "(ChemicalShifts * Isochromats)."
            )

        self.LatticeShape = (nx, ny, nz)

        # Default radius/height if not specified
        if radius is None:
            radius = 0.5 * min(nx, ny) * spacing
        if height is None:
            height = nz * spacing

        # Index ranges (0..nx-1 etc.), then center at 0 in real units
        x_idx = np.arange(nx, dtype=self.DTYPE)
        y_idx = np.arange(ny, dtype=self.DTYPE)
        z_idx = np.arange(nz, dtype=self.DTYPE)

        x_center = 0.5 * (nx - 1)
        y_center = 0.5 * (ny - 1)
        z_center = 0.5 * (nz - 1)

        x = (x_idx - x_center) * spacing
        y = (y_idx - y_center) * spacing
        z = (z_idx - z_center) * spacing

        # 3D meshgrid -> lattice coordinates
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # (nx, ny, nz)

        # Cylinder condition: x^2 + y^2 <= R^2, |z| <= height/2
        cyl_mask_3d = (X**2 + Y**2 <= radius**2) & (np.abs(Z) <= 0.5 * height)

        # Flatten for consistency with M flattening (C order)
        pos = np.column_stack((
            X.ravel(order='C'),
            Y.ravel(order='C'),
            Z.ravel(order='C')
        ))
        self.Positions = pos

        # Boolean mask, same ordering as flattened positions / M
        self.CylinderMask = cyl_mask_3d.ravel(order='C')

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

            # Mean-field part
            Wdx1, Wdy1, Wdz1 = self.Mean_DipolarField(Mx, My, Mz)

            # Secular tensor part
            Wdx2, Wdy2, Wdz2 = self.Secular_dipolar_field(Mx, My, Mz)

            # Pairwise classical dipolar field (arrays)
            Wdx3, Wdy3, Wdz3 = self.Pairwise_DipolarField(Mx, My, Mz)

            # FFT-based dipolar field on a regular lattice (if enabled)
            Wdx4, Wdy4, Wdz4 = self.FFT_DipolarField(Mx, My, Mz)

            # Total dipolar contributions (arrays)
            # Total dipolar contributions (arrays)
            Wdx = Wdx1 + Wdx2 + Wdx3 + Wdx4
            Wdy = Wdy1 + Wdy2 + Wdy3 + Wdy4
            Wdz = Wdz1 + Wdz2 + Wdz3 + Wdz4           

            Wx = Omega_X + omega_RD.real + B1_Field.real + Wdx
            Wy = Omega_Y + omega_RD.imag + B1_Field.imag + Wdy
            Wz = Omega_Z + Wdz

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

    def Plot_Lattice(self, elev=20, azim=30, figsize=(8, 8), color_by='z'):
        """
        Visualize the spin positions stored in self.Positions as a 3D scatter plot.

        Parameters
        ----------
        elev, azim : float
            Elevation and azimuth angles for 3D view.
        figsize : tuple
            Figure size passed to plt.figure.
        color_by : {'z', 'index', None}
            How to color the points:
                'z'      -> color by z-coordinate
                'index'  -> color by spin index
                None     -> single color
        """
        if self.Positions is None:
            raise ValueError("self.Positions is None. Build geometry first (line, lattice, sphere, cylinder, ...)")

        pos = np.asarray(self.Positions, dtype=self.DTYPE)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("self.Positions must have shape (N, 3).")

        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        N = pos.shape[0]

        # Choose coloring
        if color_by == 'z':
            c = z
        elif color_by == 'index':
            c = np.arange(N)
        else:
            c = 'b'  # single color

        fig = plt.figure(self.fig_counter, figsize=figsize)
        self.fig_counter += 1
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(x, y, z, c=c, s=10)

        if color_by in ('z', 'index'):
            fig.colorbar(sc, ax=ax, shrink=0.7, label=color_by)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        ax.set_title('Spin Lattice / Positions', fontsize=14)

        # Make axes equal
        max_range = np.array([x.max()-x.min(),
                              y.max()-y.min(),
                              z.max()-z.min()]).max() / 2.0
        mid_x = 0.5*(x.max()+x.min())
        mid_y = 0.5*(y.max()+y.min())
        mid_z = 0.5*(z.max()+z.min())

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, linestyle='-.')

        plt.tight_layout()
        plt.show()


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