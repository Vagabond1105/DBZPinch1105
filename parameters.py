"""
parameters.py

Two parameter categories:
- InitParams: geometry/resolution/particle-count settings, fixed once
  the sim starts running (changing these requires a restart).
- RTParams: physics knobs the user can adjust live, during a run
  (axial_current, resistivity, viscosity, etc).

Both are built from a shared Parameter class that carries its own
default/min/max and enforces int-vs-float typing on every update, so
the UI sliders and the solver always read a value that's already valid.
"""

from typing import Union

Number = Union[int, float]


class Parameter:
    """
    A single tunable value with bounds and a type.

    param_type is "init" or "rt" - purely descriptive, used by the UI to
    decide which slider group a parameter belongs in and whether it's
    still editable once sim_status != "init". Parameter itself doesn't
    enforce that distinction; SimState/UI are responsible for it.
    """

    def __init__(
        self,
        default_value: Number,
        min_value: Number,
        max_value: Number,
        param_type: str,
        value_type: type,
    ):
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.param_type = param_type
        self.value_type = value_type

        self.value_rn = self.value_type(default_value)

    def set_default(self) -> None:
        self.value_rn = self.value_type(self.default_value)

    def check_in_range(self, value: Number) -> bool:
        return self.min_value <= value <= self.max_value

    def update_value(self, new_value: Number) -> None:
        new_value = self.value_type(new_value)
        if not self.check_in_range(new_value):
            raise ValueError(
                f"{new_value} out of range [{self.min_value}, {self.max_value}]"
            )
        self.value_rn = new_value


class InitParams:
    """
    Fixed at construction. Changing any of these mid-run means the sim
    needs to be rebuilt from scratch (grid shape, particle count, and
    initial column geometry all depend on these).
    """

    def __init__(self):
        # Grid resolution - large and fixed per the architecture spec.
        # Bounds are wide (1-1000) mostly so a bad slider drag can't
        # silently produce a zero-size or negative grid; 240/240/400
        # are the intended working defaults, not a range meant to be
        # explored freely given the performance cost of this grid size.
        self.n_cells_x = Parameter(240, 1, 1000, "init", int)
        self.n_cells_y = Parameter(240, 1, 1000, "init", int)
        self.n_cells_z = Parameter(400, 1, 1000, "init", int)

        self.cell_length = Parameter(1, 1, 10, "init", int)

        self.n_particles = Parameter(20_000, 1_000, 100_000, "init", int)

        # Plasma column geometry - radius and half-length of the initial
        # cylinder. simstate.py derives core_mask from these.
        self.cylinder_radius = Parameter(60.0, 30.0, 90.0, "init", float)
        self.cylinder_length = Parameter(200.0, 100.0, 300.0, "init", float)

        # Coefficient of restitution for tracer bounce on x/y walls.
        self.e_coeff = Parameter(0.7, 0.0, 1.0, "init", float)


class RTParams:
    """
    Live-adjustable during a run. These never touch grid arrays
    directly - solvers.py reads value_rn each step and uses it as a
    source term or coefficient in the update equations.
    """

    def __init__(self):
        # Scaled by i_0 (constants.py) when used - this is a multiplier
        # on the reference current, not an absolute value.
        self.axial_current = Parameter(0.0, -20.0, 20.0, "rt", float)

        self.temp_ext = Parameter(0.0, 0.0, 1.0, "rt", float)
        self.pressure_ext = Parameter(0.0, 0.0, 1.0, "rt", float)

        # NOTE: architecture doc says "0 to 100", reference code uses a
        # min of 1.0. Kept min=1.0 here to match the working reference;
        # flag if you actually want gravity to be fully disableable (min=0).
        self.grav = Parameter(0.0, 0.0, 100.0, "rt", float)

        self.resistivity = Parameter(0.01, 0.001, 0.25, "rt", float)
        self.viscosity = Parameter(0.01, 0.0, 0.1, "rt", float)