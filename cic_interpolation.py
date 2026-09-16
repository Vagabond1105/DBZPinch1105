"""
cic_interpolation.py

Cloud-in-cell interpolation: samples Eulerian grid fields at the continuous
positions of the Lagrangian tracer particles. Per spec, this is split into
two steps rather than one combined function:
    1) cic_weights - given a position, find the 8 surrounding corner
       weights and the base grid index.
    2) cic_apply   - given those weights and a field, sum the 8 weighted
       corner values into the interpolated result.
cic_interpolate() is a thin convenience wrapper chaining the two for the
common case (used by solvers.py's tracer advection step).
"""

import numpy as np


def cic_weights(x, y, z, dx, dy, dz, origin):
    """
    Computes the 8 cloud-in-cell corner weights for a single particle
    position, plus the base grid index (i, j, k) of its lower corner.

    Weight naming is (x, y, z) order, n = negative/lower side of that
    axis's cell edge, p = positive/upper side - e.g. wpnp means +x, -y,
    +z corner. (The spec's own naming order was inconsistent between
    axes; this keeps a single consistent convention instead of copying
    that inconsistency forward.)

    Returns:
        (i, j, k), (wnnn, wpnn, wnpn, wppn, wnnp, wpnp, wnpp, wppp)
    """
    ox, oy, oz = origin

    gx = (x - ox) / dx
    gy = (y - oy) / dy
    gz = (z - oz) / dz

    i = int(np.floor(gx))
    j = int(np.floor(gy))
    k = int(np.floor(gz))

    fx = gx - i
    fy = gy - j
    fz = gz - k

    wxn, wxp = 1.0 - fx, fx
    wyn, wyp = 1.0 - fy, fy
    wzn, wzp = 1.0 - fz, fz

    wnnn = wxn * wyn * wzn
    wpnn = wxp * wyn * wzn
    wnpn = wxn * wyp * wzn
    wppn = wxp * wyp * wzn
    wnnp = wxn * wyn * wzp
    wpnp = wxp * wyn * wzp
    wnpp = wxn * wyp * wzp
    wppp = wxp * wyp * wzp

    return (i, j, k), (wnnn, wpnn, wnpn, wppn, wnnp, wpnp, wnpp, wppp)


def cic_apply(field, i, j, k, weights):
    """
    Applies the 8 CIC weights to a field (scalar (nx,ny,nz) or vector
    (nx,ny,nz,3)) around base index (i, j, k). Returns the interpolated
    value. Particles whose base cell falls outside the field's valid
    interior (needs i,i+1 both in range, etc.) return zero rather than
    wrapping or extrapolating - matches the reference's boundary guard.
    """
    wnnn, wpnn, wnpn, wppn, wnnp, wpnp, wnpp, wppp = weights

    nx, ny, nz = field.shape[:3]
    zero = 0.0 if field.ndim == 3 else np.zeros(field.shape[-1])

    if i < 0 or i >= nx - 1 or j < 0 or j >= ny - 1 or k < 0 or k >= nz - 1:
        return zero

    result = zero
    result = result + field[i,   j,   k]   * wnnn
    result = result + field[i+1, j,   k]   * wpnn
    result = result + field[i,   j+1, k]   * wnpn
    result = result + field[i+1, j+1, k]   * wppn
    result = result + field[i,   j,   k+1] * wnnp
    result = result + field[i+1, j,   k+1] * wpnp
    result = result + field[i,   j+1, k+1] * wnpp
    result = result + field[i+1, j+1, k+1] * wppp

    return result


def cic_interpolate(position, field, dx, dy, dz, origin):
    """Convenience wrapper: weights + apply in one call for one particle."""
    x, y, z = position
    (i, j, k), weights = cic_weights(x, y, z, dx, dy, dz, origin)
    return cic_apply(field, i, j, k, weights)