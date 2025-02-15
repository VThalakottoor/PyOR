# PyOR (Python on Resonance)
version: Jeener.0.9.0
## Everybody can simulate NMR Spin Physics
Author: Vineeth Francis THALAKOTTOOR JOSE CHACKO (vineethfrancis.physics@gmail.com)
<img src="https://github.com/VThalakottoor/PyOR_vJeener.0.9.0/blob/main/logo/PyOR_logo.png" alt="PyOR logo" width="300" height="150">

## 

## Why develop PyOR when many NMR simulation packages are available? Why reinvent the wheel?
The short answer: **"The Pleasure of Finding Things Out"**.

## Why Python??
The answer is: it's free.

## PyOR is for?
PyOR will be helpful for beginners (with basic knowledge of matrices, spin operators, and Python programming) interested in programming magnetic resonance pulse sequences and relaxation mechanics. It can be used to teach undergraduates and graduates about NMR.

## Main features implemented in PyOR
- Spin operators (Sx, Sy, Sz, S+ and S-) for any number of particles with any spin quantum number.
- Hamiltonians: Zeeman (lab and rotating frame), B1, J coupling, Dipolar.
- Superoperators
- Various ways to plot Matrix and expectation values.
- Time Evolution of Density Matrix
  - Evolution in Hilbert Space
    - Unitary Propagation
    - Solve ODEs
    - Radiation Damping, Raser/Maser (single and multi-mode) - Removed from the beta version, will reappear in version 1.
    - Relaxation Mechanisms (Redfield)
      - Phenomenological
      - Auto-correlated Random Field Fluctuation
      - Auto-correlated Dipolar Homonuclear
      - Auto-correlated Dipolar Heteronuclear
  - Evolution in Liouville Space
    - Unitary Propagation
    - Solve ODEs
    - Relaxation Mechanisms (Redfield)
      - Phenomenological
      - Auto-correlated Random Field Fluctuation
      - Auto-correlated Dipolar Homonuclear
      - Auto-correlated Dipolar Heteronuclear
      - Cross Correlated CSA - Dipolar Hetronuclear
      - Redfield Kite (Redkite)
    - Relaxation (Lindbladian dissipator and thermally corrected spectral density functions)
      - Auto-correlated Dipolar Homonuclear
      - Auto-correlated Dipolar Hetronuclear
