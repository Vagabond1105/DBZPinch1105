"""
solvers.py

Physics core for DBZPinch1105: vector calculus helpers, the resistive MHD
evolution step, Option C current-drive relaxation, boundary conditions,
tracer advection with wall bounce, and diagnostic bookkeeping.

Requires three constants that are NOT yet in constants.py - see the note
above compute_eta/compute_nu and above the Option C section below for why
each is needed:
    viscosity_exp        = 2.5     (opposite sign from resistivity_exp)
    transport_coeff_max  = 10.0    (provisional clip ceiling - tune later)
    ampere_relax_rate     = 1.0    (provisional - tune later)
"""

import numpy as np

from constants import (
    gamma,
    mu_0,
    rho_min,
    energy_min,
    pressure_min,
    cfl_constant,
    dt_min,
    dt_max,
    eps,
    resistivity_exp,
    viscosity_exp,        # NEW - needs adding to constants.py
    transport_coeff_max,  # NEW - needs adding to constants.py
    ampere_relax_rate,    # NEW - needs adding to constants.py
)
from cic_interpolation import cic_interpolate


# =====================================================
# vector calculus (central difference, interior-only)
# =====================================================

def gradient(f, dx):
    grad = np.zeros(f.shape + (3,), dtype=f.dtype)
    grad[1:-1, :, :, 0] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dx)
    grad[:, 1:-1, :, 1] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dx)
    grad[:, :, 1:-1, 2] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dx)
    return grad


def divergence(F, dx):
    div = np.zeros(F.shape[:-1], dtype=F.dtype)
    div[1:-1, :, :] += (F[2:, :, :, 0] - F[:-2, :, :, 0]) / (2.0 * dx)
    div[:, 1:-1, :] += (F[:, 2:, :, 1] - F[:, :-2, :, 1]) / (2.0 * dx)
    div[:, :, 1:-1] += (F[:, :, 2:, 2] - F[:, :, :-2, 2]) / (2.0 * dx)
    return div


def curl(F, dx):
    C = np.zeros_like(F)

    dFz_dy = (F[:, 2:, :, 2] - F[:, :-2, :, 2]) / (2.0 * dx)
    dFy_dz = (F[:, :, 2:, 1] - F[:, :, :-2, 1]) / (2.0 * dx)
    dFx_dz = (F[:, :, 2:, 0] - F[:, :, :-2, 0]) / (2.0 * dx)
    dFz_dx = (F[2:, :, :, 2] - F[:-2, :, :, 2]) / (2.0 * dx)
    dFy_dx = (F[2:, :, :, 1] - F[:-2, :, :, 1]) / (2.0 * dx)
    dFx_dy = (F[:, 2:, :, 0] - F[:, :-2, :, 0]) / (2.0 * dx)

    C[1:-1, 1:-1, 1:-1, 0] = dFz_dy[1:-1, :, 1:-1] - dFy_dz[1:-1, 1:-1, :]
    C[1:-1, 1:-1, 1:-1, 1] = dFx_dz[1:-1, 1:-1, :] - dFz_dx[:, 1:-1, 1:-1]
    C[1:-1, 1:-1, 1:-1, 2] = dFy_dx[:, 1:-1, 1:-1] - dFx_dy[1:-1, :, 1:-1]

    return C


def laplacian(F, dx):
    L = np.zeros_like(F)

    if F.ndim == 3:
        L[1:-1, 1:-1, 1:-1] = (
            F[2:, 1:-1, 1:-1] + F[:-2, 1:-1, 1:-1]
            + F[1:-1, 2:, 1:-1] + F[1:-1, :-2, 1:-1]
            + F[1:-1, 1:-1, 2:] + F[1:-1, 1:-1, :-2]
            - 6.0 * F[1:-1, 1:-1, 1:-1]
        ) / (dx * dx)
        return L

    for a in range(F.shape[-1]):
        Fa = F[..., a]
        La = np.zeros_like(Fa)
        La[1:-1, 1:-1, 1:-1] = (
            Fa[2:, 1:-1, 1:-1] + Fa[:-2, 1:-1, 1:-1]
            + Fa[1:-1, 2:, 1:-1] + Fa[1:-1, :-2, 1:-1]
            + Fa[1:-1, 1:-1, 2:] + Fa[1:-1, 1:-1, :-2]
            - 6.0 * Fa[1:-1, 1:-1, 1:-1]
        ) / (dx * dx)
        L[..., a] = La

    return L


# =====================================================
# transport coefficients
# =====================================================
# eta (resistivity) and nu (viscosity) both use a temperature power law,
# but they need OPPOSITE-signed exponents - this was a real bug in the
# GPT reference, which used the same T^-1.5 for both. Spitzer resistivity
# genuinely falls as T^-3/2 (hotter plasma conducts better). Classical
# (Braginskii) viscosity does the opposite, rising roughly as T^+5/2
# (hotter plasma has a longer mean free path and diffuses momentum more
# readily). viscosity_exp = +2.5 is a new constant needed in constants.py.
#
# Both are also hard-clipped at transport_coeff_max. In near-vacuum cells
# the temperature floor still leaves T^-1.5 large enough to blow up eta
# and silently collapse dt via the diffusive cap - this was flagged
# earlier and is fixed here instead of deferred.

def compute_eta(T, eta0):
    eta = eta0 * (T ** resistivity_exp)
    return np.clip(eta, 0.0, transport_coeff_max)


def compute_nu(T, nu0):
    nu = nu0 * (T ** viscosity_exp)
    return np.clip(nu, 0.0, transport_coeff_max)


def viscous_heating(v, nu, dx):
    """
    Viscous dissipation rate, added to the energy equation. This term was
    flagged as MISSING entirely from the original spec/reference - without
    it, viscosity removes kinetic energy in the momentum equation but that
    energy just vanishes instead of becoming heat, which quietly breaks
    energy conservation.

    This is NOT the full compressible viscous stress tensor (that needs
    the complete, trace-free strain-rate tensor from all 9 velocity
    gradient components). It's a positive-definite proxy,
    nu * sum_a |grad(v_a)|^2, that grows with velocity shear the same way
    real dissipation does, built only from the vector calculus already
    defined above. Reasonable for the physics-validation-on-a-small-grid
    stage; revisit if energy-conservation checks want tighter accuracy.
    """
    heating = np.zeros(v.shape[:-1])
    for a in range(3):
        grad_va = gradient(v[..., a], dx)
        heating += np.sum(grad_va * grad_va, axis=-1)
    return nu * heating


# =====================================================
# timestep selection (hyperbolic + diffusive + dedner)
# =====================================================

def compute_dt(dx, v_fluid, c_s, v_a, eta, nu):
    """
    dt is capped by three independent stability constraints, then clamped
    to [dt_min, dt_max]:
      - hyperbolic CFL, including the Dedner cleaning speed ch as an
        additional signal speed (see note in dedner_clean - this was
        previously computed independently of dt, which was flagged as a
        real inconsistency; ch is now derived from and fed back into the
        same CFL constraint).
      - explicit resistive diffusion
      - explicit viscous diffusion

    Returns (dt, ch) - ch is reused by dedner_clean so the two stay
    coupled to a single value each step.
    """
    v_max_physical = v_fluid + c_s + v_a
    ch = float(np.max(v_max_physical))
    v_max_total = np.maximum(v_max_physical + ch, eps)

    dt_hyp = cfl_constant * float(np.min(dx / v_max_total))

    eta_max = float(np.max(eta))
    nu_max = float(np.max(nu))
    dt_eta = 0.2 * dx * dx / (eta_max + eps)
    dt_nu = 0.2 * dx * dx / (nu_max + eps)

    dt = min(dt_hyp, dt_eta, dt_nu)
    return float(np.clip(dt, dt_min, dt_max)), ch


# =====================================================
# boundary conditions - fluid
# =====================================================

def reflect_xy(sim_state):
    """
    Reflective (bounce) boundaries in x and y for the FLUID, damped by
    e_coeff. e_coeff's own doc says it's "for the bounce on walls ... and
    damps for the fluid" but the reference never actually wired that
    damping in anywhere - it flipped momentum sign with no damping at
    all. Applied here as the coefficient of restitution on reflection.
    """
    e = sim_state.e_coeff

    rho = sim_state.rho
    u = sim_state.internal_energy
    m = sim_state.momentum
    B = sim_state.b_field

    # x min / max
    rho[0, :, :] = rho[1, :, :]
    u[0, :, :] = u[1, :, :]
    m[0, :, :, 0] = -e * m[1, :, :, 0]
    m[0, :, :, 1] = m[1, :, :, 1]
    m[0, :, :, 2] = m[1, :, :, 2]
    B[0, :, :, 0] = -B[1, :, :, 0]
    B[0, :, :, 1] = B[1, :, :, 1]
    B[0, :, :, 2] = B[1, :, :, 2]

    rho[-1, :, :] = rho[-2, :, :]
    u[-1, :, :] = u[-2, :, :]
    m[-1, :, :, 0] = -e * m[-2, :, :, 0]
    m[-1, :, :, 1] = m[-2, :, :, 1]
    m[-1, :, :, 2] = m[-2, :, :, 2]
    B[-1, :, :, 0] = -B[-2, :, :, 0]
    B[-1, :, :, 1] = B[-2, :, :, 1]
    B[-1, :, :, 2] = B[-2, :, :, 2]

    # y min / max
    rho[:, 0, :] = rho[:, 1, :]
    u[:, 0, :] = u[:, 1, :]
    m[:, 0, :, 0] = m[:, 1, :, 0]
    m[:, 0, :, 1] = -e * m[:, 1, :, 1]
    m[:, 0, :, 2] = m[:, 1, :, 2]
    B[:, 0, :, 0] = B[:, 1, :, 0]
    B[:, 0, :, 1] = -B[:, 1, :, 1]
    B[:, 0, :, 2] = B[:, 1, :, 2]

    rho[:, -1, :] = rho[:, -2, :]
    u[:, -1, :] = u[:, -2, :]
    m[:, -1, :, 0] = m[:, -2, :, 0]
    m[:, -1, :, 1] = -e * m[:, -2, :, 1]
    m[:, -1, :, 2] = m[:, -2, :, 2]
    B[:, -1, :, 0] = B[:, -2, :, 0]
    B[:, -1, :, 1] = -B[:, -2, :, 1]
    B[:, -1, :, 2] = B[:, -2, :, 2]


def outflow_z(sim_state):
    """Outflow (zero-gradient copy) boundary for the FLUID in z."""
    rho = sim_state.rho
    u = sim_state.internal_energy
    m = sim_state.momentum
    B = sim_state.b_field

    rho[:, :, 0] = rho[:, :, 1]
    u[:, :, 0] = u[:, :, 1]
    m[:, :, 0, :] = m[:, :, 1, :]
    B[:, :, 0, :] = B[:, :, 1, :]

    rho[:, :, -1] = rho[:, :, -2]
    u[:, :, -1] = u[:, :, -2]
    m[:, :, -1, :] = m[:, :, -2, :]
    B[:, :, -1, :] = B[:, :, -2, :]


def apply_boundaries(sim_state):
    reflect_xy(sim_state)
    outflow_z(sim_state)


# =====================================================
# boundary conditions - tracers
# =====================================================

def bounce_tracers_xy(sim_state):
    """
    Tracer wall-bounce in x/y. Flagged previously as UNIMPLEMENTED - the
    reference only ever removed tracers (in z), it never bounced them,
    even though Sim_Architecture.md explicitly says tracers "bounce off
    walls if they hit boundaries." Mirrors any tracer past a wall back
    into the domain and damps its stored velocity by e_coeff, matching
    the fluid's own reflective boundary treatment above.
    """
    pos = sim_state.particle_positions
    vel = sim_state.particle_velocities
    if pos.size == 0:
        return

    e = sim_state.e_coeff
    x_max = sim_state.lx / 2.0
    y_max = sim_state.ly / 2.0

    over_xp = pos[:, 0] > x_max
    pos[over_xp, 0] = 2.0 * x_max - pos[over_xp, 0]
    vel[over_xp, 0] *= -e

    over_xn = pos[:, 0] < -x_max
    pos[over_xn, 0] = -2.0 * x_max - pos[over_xn, 0]
    vel[over_xn, 0] *= -e

    over_yp = pos[:, 1] > y_max
    pos[over_yp, 1] = 2.0 * y_max - pos[over_yp, 1]
    vel[over_yp, 1] *= -e

    over_yn = pos[:, 1] < -y_max
    pos[over_yn, 1] = -2.0 * y_max - pos[over_yn, 1]
    vel[over_yn, 1] *= -e


def remove_exited_tracers(sim_state):
    """Tracers that leave axially (z) are removed - outflow, not bounce."""
    pos = sim_state.particle_positions
    vel = sim_state.particle_velocities
    if pos.size == 0:
        return

    z_min = -sim_state.lz / 2.0
    z_max = sim_state.lz / 2.0
    mask = (pos[:, 2] > z_min) & (pos[:, 2] < z_max)

    sim_state.particle_positions = pos[mask]
    sim_state.particle_velocities = vel[mask]
    sim_state.n_particles = sim_state.particle_positions.shape[0]


# =====================================================
# dedner divergence cleaning
# =====================================================

def dedner_clean(sim_state, divB, ch):
    """
    Minimal generalized-Lagrange-multiplier divergence cleaning:
        dpsi/dt = -ch^2 * divB - cp^2 * psi
        dB/dt  += -grad(psi)

    ch is passed in from compute_dt - the SAME speed already folded into
    the CFL constraint, rather than an independently-chosen value left
    decoupled from dt (previously flagged as a real inconsistency).
    """
    dx = sim_state.dx
    dt = sim_state.dt

    ch_safe = max(ch, 1e-6)
    cp = ch_safe / max(dx, 1e-12)

    sim_state.psi += dt * (-(ch_safe ** 2) * divB - (cp ** 2) * sim_state.psi)
    sim_state.b_field -= dt * gradient(sim_state.psi, dx)


# =====================================================
# core evolution step
# =====================================================

def evolve_simstate(sim_state):
    """
    One explicit timestep, in the sequence given by Sim_Architecture.md:
      1) derive contingent fields from the current core state
      2) compute dt via CFL
      3) update rho
      4) update momentum
      5) update internal_energy
      6) update b_field (induction + resistive diffusion + Option C drive)
      7) apply fluid boundary conditions
      8) advect tracers, then apply tracer boundary conditions
      9) update diagnostics
      10) advance time

    NOTE: temp_ext and pressure_ext (RTParams) are not wired into any
    equation here. Nothing in the solvers spec ever showed how they should
    enter the physics - this looks like a genuine gap in the original
    spec rather than something implicit. Flagging rather than inventing
    an interpretation; not load-bearing for the first physics-validation
    pass on the small grid.
    """

    dx = sim_state.dx

    # ---------------------------------------------------
    # 1) contingent fields from current core state
    # ---------------------------------------------------
    v = sim_state.velocity()
    p = sim_state.pressure()
    T = sim_state.temperature()

    B = sim_state.b_field
    rho = np.maximum(sim_state.rho, rho_min)

    c_s = sim_state.sound_speed()
    v_a = sim_state.alfven_speed()
    v_fluid = np.linalg.norm(v, axis=-1)

    J = curl(B, dx) / mu_0
    J2 = np.sum(J * J, axis=-1)

    eta0 = float(sim_state.rt_params.resistivity.value_rn)
    nu0 = float(sim_state.rt_params.viscosity.value_rn)
    eta = compute_eta(T, eta0)
    nu = compute_nu(T, nu0)

    gravity_vec = sim_state.gravity_vector()

    # ---------------------------------------------------
    # 2) dt via cfl (+ diffusive caps, + dedner speed)
    # ---------------------------------------------------
    sim_state.dt, ch = compute_dt(dx, v_fluid, c_s, v_a, eta, nu)
    dt = sim_state.dt

    # ---------------------------------------------------
    # 3) update rho
    # ---------------------------------------------------
    sim_state.rho -= dt * divergence(sim_state.rho[..., None] * v, dx)
    sim_state.rho = np.maximum(sim_state.rho, rho_min)
    rho = np.maximum(sim_state.rho, rho_min)

    # ---------------------------------------------------
    # 4) update momentum
    # ---------------------------------------------------
    m = sim_state.momentum
    for a in range(3):
        flux = m[..., a][..., None] * v
        m[..., a] -= dt * divergence(flux, dx)

    gradP = gradient(p, dx)
    JxB = np.cross(J, B, axis=-1)
    visc_force = nu[..., None] * laplacian(v, dx)
    grav_force = rho[..., None] * gravity_vec[None, None, None, :]

    m += dt * (-gradP + JxB + grav_force + visc_force)

    vac = sim_state.rho <= rho_min
    m[vac] = 0.0

    # ---------------------------------------------------
    # 5) update internal_energy
    # ---------------------------------------------------
    u = sim_state.internal_energy
    u -= dt * divergence(u[..., None] * v, dx)

    divv = divergence(v, dx)
    ohmic = eta * J2
    visc_heat = viscous_heating(v, nu, dx)  # previously missing entirely

    u += dt * (-p * divv + ohmic + visc_heat)
    sim_state.internal_energy = np.maximum(u, energy_min)

    # ---------------------------------------------------
    # 6) update b_field: induction + resistive diffusion
    #    + Option C current-drive relaxation (locked design decision)
    # ---------------------------------------------------
    induction = curl(np.cross(v, B, axis=-1), dx)
    resistive_diffusion = curl(eta[..., None] * J, dx)

    b_target = sim_state.ampere_bfield_target()
    current_drive = ampere_relax_rate * (b_target - sim_state.b_field)

    sim_state.b_field += dt * (induction - resistive_diffusion + current_drive)

    divB = divergence(sim_state.b_field, dx)
    dedner_clean(sim_state, divB, ch)

    # ---------------------------------------------------
    # 7) fluid boundary conditions
    # ---------------------------------------------------
    sim_state.rho = np.maximum(sim_state.rho, rho_min)
    sim_state.internal_energy = np.maximum(sim_state.internal_energy, energy_min)
    apply_boundaries(sim_state)

    # ---------------------------------------------------
    # 8) advect tracers, then tracer boundary conditions
    # ---------------------------------------------------
    v_current = sim_state.velocity()
    origin = (-sim_state.lx / 2.0, -sim_state.ly / 2.0, -sim_state.lz / 2.0)

    pos = sim_state.particle_positions
    if pos.size != 0:
        v_tracer = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            v_tracer[i] = cic_interpolate(
                pos[i], v_current, sim_state.dx, sim_state.dy, sim_state.dz, origin
            )
        sim_state.particle_velocities = v_tracer
        sim_state.particle_positions = pos + dt * v_tracer

    bounce_tracers_xy(sim_state)
    remove_exited_tracers(sim_state)

    # ---------------------------------------------------
    # 9) diagnostics - lists always exist on sim_state already,
    #    no hasattr checks needed
    # ---------------------------------------------------
    ke = 0.5 * sim_state.rho * np.sum(v_current * v_current, axis=-1)
    me = 0.5 * sim_state.b2() / mu_0
    ie = sim_state.internal_energy

    total_ke = float(np.sum(ke))
    total_me = float(np.sum(me))
    total_ie = float(np.sum(ie))
    total_e = total_ke + total_me + total_ie

    sim_state.kinetic_energy_history.append(total_ke)
    sim_state.magnetic_energy_history.append(total_me)
    sim_state.internal_energy_history.append(total_ie)
    sim_state.total_energy_history.append(total_e)

    p_now = sim_state.pressure()
    T_now = sim_state.temperature()
    core = sim_state.core_mask

    avg_rho = float(np.mean(sim_state.rho[core]))
    avg_p = float(np.mean(p_now[core]))
    avg_T = float(np.mean(T_now[core]))

    sim_state.core_density_evolution.append(avg_rho)
    sim_state.core_pressure_evolution.append(avg_p)
    sim_state.core_temperature_evolution.append(avg_T)

    total_ohmic = float(np.sum(eta * J2))
    tau_conf = total_ie / (total_ohmic + eps)
    sim_state.confinement_evolution.append(tau_conf)

    lawson_proxy = avg_rho * avg_T * tau_conf
    sim_state.lawson_evolution.append(lawson_proxy)

    sim_state.time_array.append(float(sim_state.t))

    # ---------------------------------------------------
    # 10) advance time
    # ---------------------------------------------------
    sim_state.t += dt
    sim_state.n_dt += 1