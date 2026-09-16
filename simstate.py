"""
simstate.py

SimState holds the full state of the DBZPinch1105 simulation at a single
instant in time: global run data, the Eulerian grid fields, the Lagrangian
tracer particles, and the diagnostic history arrays. This module holds DATA
ONLY - no evolution algorithms live here (see solvers.py for that).
"""

import numpy as np

from constants import (
    gamma,
    mu_0,
    rho_0,
    pressure_0,
    rho_min,
    energy_min,
    temp_min,
    pressure_min,
    cfl_constant,
    dt_min,
    dt_max,
    eps,
    i_0,
)
from parameters import InitParams, RTParams


class SimState:

    def __init__(self, init_params: InitParams, rt_params: RTParams):

        # =====================================================
        # global runtime data
        # =====================================================

        self.t = 0.0
        self.dt = dt_min
        self.n_dt = 0

        self.sim_status = "init"  # "init", "run", "pause"
        self.bounce = True

        self.init_params = init_params
        self.rt_params = rt_params

        # =====================================================
        # grid dimensions
        # =====================================================

        nx = init_params.n_cells_x.value_rn
        ny = init_params.n_cells_y.value_rn
        nz = init_params.n_cells_z.value_rn
        self.nx, self.ny, self.nz = nx, ny, nz

        dx = init_params.cell_length.value_rn
        dy = dx
        dz = dx
        self.dx, self.dy, self.dz = dx, dy, dz

        self.lx = nx * dx
        self.ly = ny * dy
        self.lz = nz * dz

        self.x0, self.y0, self.z0 = 0.0, 0.0, 0.0

        # =====================================================
        # coordinate grids (geometry / masks only, not solved on)
        # =====================================================

        x = (np.arange(nx) - nx / 2.0) * dx
        y = (np.arange(ny) - ny / 2.0) * dy
        z = (np.arange(nz) - nz / 2.0) * dz

        self.x_coords, self.y_coords, self.z_coords = x, y, z
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")

        # =====================================================
        # eulerian fields
        # =====================================================

        self.rho = np.zeros((nx, ny, nz))
        self.internal_energy = np.zeros((nx, ny, nz))
        self.momentum = np.zeros((nx, ny, nz, 3))
        self.b_field = np.zeros((nx, ny, nz, 3))

        # scalar potential for Dedner divergence cleaning (see solvers.py).
        # Allocated here rather than lazily via hasattr in solvers.py, to
        # stay consistent with the "no dynamically-added attributes" rule
        # already applied to the diagnostic history lists below.
        self.psi = np.zeros((nx, ny, nz))

        # =====================================================
        # plasma column geometry + core_mask
        # =====================================================

        self.cylinder_radius = init_params.cylinder_radius.value_rn
        self.cylinder_length = init_params.cylinder_length.value_rn

        # RESOLVED spec conflict (previously flagged): Abstract-README and
        # the sim_state.py.md bullet list literally give
        #   r_core = 0.5 * cylinder_radius
        #   z_core_half = 0.25 * cylinder_length
        # but cylinder_radius/cylinder_length are documented as the column's
        # actual dimensions, and tracers are seeded across the FULL column
        # (see below) and described as being sprinkled "through the
        # plasma". Shrinking core_mask to a quarter of the column would
        # contradict both of those. Treating the 0.5/0.25 formula as the
        # actual error and using the full column dimensions instead.
        r_core = self.cylinder_radius
        z_core_half = 0.5 * self.cylinder_length

        self.core_mask = (
            (self.X ** 2 + self.Y ** 2 < r_core ** 2)
            & (np.abs(self.Z) < z_core_half)
        )

        self.r_core = r_core
        self.z_core_half = z_core_half

        # seed plasma column vs. vacuum floor
        internal_energy_plasma = pressure_0 / (gamma - 1.0)

        self.rho[self.core_mask] = rho_0
        self.rho[~self.core_mask] = rho_min

        self.internal_energy[self.core_mask] = internal_energy_plasma
        self.internal_energy[~self.core_mask] = energy_min

        # boundary restitution coefficient (used by solvers for tracer bounce)
        self.e_coeff = init_params.e_coeff.value_rn

        # =====================================================
        # lagrangian tracer particles
        # =====================================================
        # NOTE: tracer seeding uses self.cylinder_radius / cylinder_length
        # directly (same full dimensions as core_mask now uses). Kept as
        # separate variables rather than reusing r_core/z_core_half so the
        # two seeding steps stay decoupled if core_mask's definition ever
        # changes again.

        n_target = init_params.n_particles.value_rn

        x_extent = 2.0 * self.cylinder_radius
        y_extent = 2.0 * self.cylinder_radius
        z_extent = self.cylinder_length

        n_prime = int((4.0 / np.pi) * n_target)

        d_spacing = (x_extent * y_extent * z_extent / n_prime) ** (1.0 / 3.0)

        xp = np.arange(-x_extent / 2.0, x_extent / 2.0, d_spacing)
        yp = np.arange(-y_extent / 2.0, y_extent / 2.0, d_spacing)
        zp = np.arange(-z_extent / 2.0, z_extent / 2.0, d_spacing)

        Xp, Yp, Zp = np.meshgrid(xp, yp, zp, indexing="ij")
        lattice = np.vstack([Xp.ravel(), Yp.ravel(), Zp.ravel()]).T

        cyl_mask = (
            lattice[:, 0] ** 2 + lattice[:, 1] ** 2 <= self.cylinder_radius ** 2
        )
        lattice = lattice[cyl_mask]

        self.particle_positions = lattice[:n_target].copy()
        self.particle_velocities = np.zeros_like(self.particle_positions)

        # actual tracer count after lattice culling can differ slightly
        # from the requested n_particles - store what we actually got
        self.n_particles = self.particle_positions.shape[0]

        # =====================================================
        # diagnostic history - initialised as empty lists up front,
        # never added dynamically via hasattr() checks in solvers.py
        # =====================================================

        self.kinetic_energy_history = []
        self.magnetic_energy_history = []
        self.internal_energy_history = []
        self.total_energy_history = []

        self.core_density_evolution = []
        self.core_pressure_evolution = []
        self.core_temperature_evolution = []

        self.confinement_evolution = []
        self.lawson_evolution = []

        self.time_array = []

    # =====================================================
    # derived fields
    # =====================================================
    # kept to the simple, purely-local derived quantities here.
    # current_density and resistivity depend on curl(B), which is a
    # vector-calculus operator that belongs to solvers.py (avoids a
    # circular import and keeps "data vs algorithm" separation clean).

    def velocity(self):
        vel = np.zeros_like(self.momentum)
        mask = self.rho > rho_min
        vel[mask] = self.momentum[mask] / self.rho[mask][..., None]
        return vel

    def pressure(self):
        p = (gamma - 1.0) * self.internal_energy
        return np.maximum(p, pressure_min)

    def temperature(self):
        p = self.pressure()
        rho_safe = np.maximum(self.rho, rho_min)
        T = p / rho_safe
        return np.maximum(T, temp_min)

    def b2(self):
        return np.sum(self.b_field * self.b_field, axis=-1)

    def sound_speed(self):
        p = self.pressure()
        rho_safe = np.maximum(self.rho, rho_min)
        return np.sqrt(gamma * p / rho_safe)

    def alfven_speed(self):
        rho_safe = np.maximum(self.rho, rho_min)
        return np.sqrt(self.b2() / (rho_safe * mu_0))

    def gravity_vector(self):
        # gravity acts in the negative y direction per Sim_Architecture.md
        # (this corrects the GPT reference code, which used negative z)
        g = self.rt_params.grav.value_rn
        return np.array([0.0, -g, 0.0])

    # =====================================================
    # option c current-drive: closed-form ampere b_phi target
    # =====================================================

    def ampere_bfield_target(self):
        """
        Locked design decision (Option C): a closed-form azimuthal B_phi
        profile derived from core_mask geometry, standing in for the
        current-driven field of the Z-pinch. solvers.py relaxes b_field
        toward this target as a source term in the induction equation.

        This is NOT boundary-driven diffusion (Option A) and NOT a
        volumetric Poisson solve (Option B).

        Physics: treats the core as a cylinder of radius r_core carrying
        a uniformly distributed axial current I. Ampere's law then gives

            B_phi(r) = mu_0 * I * r / (2*pi*r_core^2)   for r <= r_core
            B_phi(r) = mu_0 * I / (2*pi*r)               for r >  r_core

        restricted to the axial extent of the core (|Z| < z_core_half) -
        outside that axial band the target field is zero, since no current
        is assumed to flow there. A pure B_phi(r) field is automatically
        divergence-free in this geometry, so no extra cleanup pass is
        required for this target field on its own.
        """

        axial_current = self.rt_params.axial_current.value_rn
        current = axial_current * i_0

        r = np.sqrt(self.X ** 2 + self.Y ** 2)
        r_safe = np.maximum(r, eps)

        b_phi_inside = mu_0 * current * r / (2.0 * np.pi * self.r_core ** 2)
        b_phi_outside = mu_0 * current / (2.0 * np.pi * r_safe)

        b_phi = np.where(r <= self.r_core, b_phi_inside, b_phi_outside)

        axial_active = np.abs(self.Z) < self.z_core_half
        b_phi = np.where(axial_active, b_phi, 0.0)

        b_target = np.zeros((self.nx, self.ny, self.nz, 3))
        b_target[..., 0] = -b_phi * self.Y / r_safe
        b_target[..., 1] = b_phi * self.X / r_safe
        # b_target[..., 2] stays 0 - purely azimuthal field

        return b_target

    # =====================================================
    # cfl condition
    # =====================================================

    def update_dt_cfl(self):
        vel = self.velocity()
        v_mag = np.linalg.norm(vel, axis=-1)

        v_max = v_mag + self.sound_speed() + self.alfven_speed()
        v_max = np.maximum(v_max, eps)

        dt = cfl_constant * np.min(self.dx / v_max)
        self.dt = float(np.clip(dt, dt_min, dt_max))