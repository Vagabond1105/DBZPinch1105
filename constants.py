"""
constants.py

Internal reference and relational-frame constants for DBZPinch1105. These
are NOT hard physics constants (like Boltzmann's constant) - they are
normalised scaling numbers that define what "1 unit" of density, length,
pressure, etc. means throughout the simulation, and a handful of numerical
safety floors/ceilings the solvers need to stay stable.
"""

import numpy as np

# =====================================================
# normalisation base units
# =====================================================
# working entirely in normalised units - these are the reference scales
# that everything else (rho, pressure, B, velocity, time) is measured
# relative to.

gamma = 5.0 / 3.0
# adiabatic index

mu_0 = 1.0
# vacuum permeability, normalised to 1

l_0 = 1.0
# reference length - one unit of length throughout the sim, geometry is
# relative to this

dx = 1.0
dy = 1.0
dz = 1.0
# default cell spacing (also exposed as InitParams.cell_length - this is
# the constants-module reference value, not a runtime override)

rho_0 = 1.0
# default/reference plasma density - what "rho = 1" means

pressure_0 = 1.0
# reference pressure - what "pressure = 1" means

b_0 = 1.0
# reference magnetic field strength - sets the scale of the Alfven speed

v_0 = b_0 / rho_0 ** 0.5
# characteristic plasma speed, used for CFL scaling

cs_0 = (gamma * pressure_0 / rho_0) ** 0.5
# reference sound speed

t_0 = l_0 / v_0
# one unit of time = one cell-crossing time at v_0; dt is relative to this

i_0 = b_0 * l_0 / mu_0
# converts a normalised axial current into a magnetic field strength
# (used directly by ampere_bfield_target in solvers.py)

# =====================================================
# transport coefficient temperature scaling
# =====================================================
# resistivity and viscosity use OPPOSITE-signed temperature exponents.
# This distinction matters - the original GPT-scaffolded reference used
# the same exponent for both, which was flagged and fixed in solvers.py.

resistivity_exp = -1.5
# Spitzer-like resistivity: eta = eta0 * T^resistivity_exp. Hotter plasma
# conducts better, so resistivity FALLS with temperature.

viscosity_exp = 2.5
# NEW (added for solvers.py) - Braginskii-like viscosity:
# nu = nu0 * T^viscosity_exp. Hotter plasma has a longer mean free path
# and diffuses momentum more readily, so viscosity RISES with
# temperature - the opposite trend from resistivity above.

transport_coeff_max = 10.0
# NEW (added for solvers.py) - PROVISIONAL clip ceiling applied to both
# eta and nu after the temperature power law. Without this, a near-floor
# temperature in a vacuum cell can still blow up T^resistivity_exp large
# enough to collapse the CFL timestep via the diffusive dt cap. Tune this
# once the 32^3 physics-validation grid is running - this value has not
# been derived from anything, just chosen to be "clearly above normal
# operating range, clearly below numerically dangerous."

# =====================================================
# option c current-drive relaxation rate
# =====================================================

ampere_relax_rate = 1.0
# NEW (added for solvers.py) - PROVISIONAL relaxation rate (1/time) at
# which b_field is pulled toward ampere_bfield_target() each step, i.e.
# the source term is ampere_relax_rate * (b_target - b_field). Higher
# values make the current-drive mechanism respond to axial_current
# changes faster but fight the natural induction/resistive evolution more
# aggressively; too high risks instability, too low makes the pinch feel
# unresponsive to the slider. Needs tuning against the validation grid,
# not derived from first principles.

# =====================================================
# camera control (ui_ux.py)
# =====================================================

cam_radius_min = 100.0
cam_radius_max = 1000.0
cam_radius_default = 500.0
# NEW (added for ui_ux.py) - ui_ux.py.md says the camera's radius_range
# and r_def come "from constants", but no such constants existed here or
# anywhere in the original spec. cam_radius_default matches the distance
# already used in visualisation.py's reference TurntableCamera(distance=
# 500); min/max are a provisional +/-400 window around that, not derived
# from anything - revisit once the grid is actually being rendered and
# you know what range feels right.

# =====================================================
# camera (ui_ux.py CameraController)
# =====================================================

cam_r_min = 50.0
cam_r_max = 2000.0
cam_r_def = 500.0
# NEW (added for ui_ux.py) - PROVISIONAL orbital-camera distance bounds
# and default. cam_r_def ~ 500 comfortably fits the full 240x240x400
# grid in view (box diagonal is ~525 at default cell_length=1), matching
# the distance already used in visualisation.py.md's VisPy reference.
# NOTE: CameraController may end up unused - see ui_ux.py's own note on
# the conflict with VisPy's built-in TurntableCamera.

# =====================================================
# numerical safety floors
# =====================================================
# prevent divide-by-zero and unphysical negative values in near-vacuum
# cells outside the plasma column.

rho_min = 1e-6
energy_min = 1e-8
temp_min = 1e-6
pressure_min = 1e-8

# =====================================================
# fusion threshold proxy
# =====================================================

fusion_threshold_proxy = 1.0
# UNPHYSICAL / PROVISIONAL - this is a placeholder scale for the Lawson
# proxy diagnostic line in exports_diagnostics.py, not a derived fusion
# condition. Flagged explicitly so it's never mistaken for a real
# physical threshold.

# =====================================================
# cfl / timestep control
# =====================================================

cfl_constant = 0.4
dt_min = 1e-6
dt_max = 1e-1

# =====================================================
# axis indices
# =====================================================

X, Y, Z = 0, 1, 2
# so array indexing can read as rho[..., X] instead of rho[..., 0]

# =====================================================
# small epsilon
# =====================================================

eps = 1e-12
# generic small number to guard against division by zero