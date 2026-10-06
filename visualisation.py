"""
visualisation.py

3D rendering of the simulation via VisPy: a density isosurface plus the
Lagrangian tracer particles, an iso-level slider, and a bridge method that
applies ui_ux.py's keyboard-driven CameraController onto VisPy's own
TurntableCamera. Render is refreshed on its own 60Hz timer, independent of
the (variable, CFL-driven) physics timestep in solvers.py.

Deviations from the literal spec (see chat for full reasoning):
- Dropped the `from skimage.measure import marching_cubes` import - it's
  never actually used (VisPy's Isosurface visual builds its own mesh
  internally) and skimage isn't even in requirements.txt.
- Added tracer particle rendering (visuals.Markers) - the spec here only
  covered the isosurface, but tracers are core to the whole project's
  stated visual purpose and were never wired into any rendering code
  anywhere in the original spec.
- Fixed the iso-slider's y-position - the reference hardcoded y=740 on an
  800-tall canvas, which (VisPy's widget-space y=0 is the top) puts it
  near the BOTTOM of the window, contradicting the spec's own "centre top
  of screen" instruction.
- set_data(data=..., level=...) in one call isn't reliable against
  VisPy's real IsosurfaceVisual API - split into two statements.
- CONFIRMED BUG (not just a hypothesized one): `from vispy.scene.widgets
  import Slider` throws ImportError on an actual run - that class doesn't
  exist in vispy.scene.widgets. Replaced with IsoLevelSlider, a small
  self-contained slider built from the same Rectangle+Ellipse visuals
  ui_ux.py's ParameterSlider already uses. main.py now needs to drive it
  via interact_mouse()/render() every tick, same as the other sliders.
"""

import numpy as np

from vispy import app, scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform

from constants import rho_min


class ZPinchVisualiser:

    def __init__(self, sim_state):

        self.sim_state = sim_state

        # ----------------------------
        # canvas & scene
        # ----------------------------
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            size=(1200, 800),
            show=True,
        )

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=500,
        )

        # ----------------------------
        # isosurface settings
        # ----------------------------
        self.iso_alpha = 0.5  # relative iso value (0-1), "medium" default
        self.scalar_field = self.sim_state.rho

        self.surface = visuals.Isosurface(
            data=self.scalar_field,
            level=self._compute_iso_value(),
            parent=self.view.scene,
            color=(0.2, 0.6, 1.0, 0.8),
        )

        # The isosurface mesh comes out of marching cubes in GRID-INDEX
        # space (0..nx, 0..ny, 0..nz) - it knows nothing about dx/dy/dz.
        # sim_state's own coordinate grids (X, Y, Z), core_mask, and the
        # tracer positions are all in PHYSICAL space (centred on 0, scaled
        # by dx/dy/dz). This transform maps the surface into that same
        # physical space so it lines up with everything else, rather than
        # a bare translate that would silently assume dx=dy=dz=1 forever.
        dx, dy, dz = sim_state.dx, sim_state.dy, sim_state.dz
        nx, ny, nz = sim_state.nx, sim_state.ny, sim_state.nz

        self.surface.transform = STTransform(
            scale=(dx, dy, dz),
            translate=(-nx * dx / 2.0, -ny * dy / 2.0, -nz * dz / 2.0),
        )

        # ----------------------------
        # tracer particles
        # ----------------------------
        # Already stored in physical coordinates (see simstate.py), so
        # unlike the isosurface above, these need no extra transform -
        # they're added directly to view.scene.
        self.tracers = visuals.Markers(parent=self.view.scene)
        self._update_tracers()

        # dirty-tracking for update() below - see there for why this
        # matters. Initialised to the values just used above so the very
        # first tick doesn't immediately redo the work __init__ just did.
        self._last_rendered_n_dt = sim_state.n_dt
        self._last_rendered_alpha = self.iso_alpha

        # ----------------------------
        # iso-level slider
        # ----------------------------
        self._create_slider()

        # ----------------------------
        # timer for 60 fps update
        # ----------------------------
        self.timer = app.Timer(
            interval=1 / 60,
            connect=self.update,
            start=True,
        )

    # ==========================================================
    # iso value
    # ==========================================================

    def _compute_iso_value(self):
        field = self.scalar_field
        raw_level = self.iso_alpha * np.max(field)
        # guard against near-vacuum degenerate isosurfaces: if the whole
        # field is close to rho_min (e.g. plasma has fully dispersed),
        # np.max(field) can sit right at the vacuum floor, and a level
        # that low would render an isosurface engulfing the entire
        # near-vacuum background instead of showing nothing meaningful.
        return max(raw_level, 2.0 * rho_min)

    # ==========================================================
    # tracers
    # ==========================================================

    def _update_tracers(self):
        positions = self.sim_state.particle_positions
        if positions.size == 0:
            self.tracers.set_data(pos=np.zeros((0, 3)))
            return

        self.tracers.set_data(
            pos=positions,
            size=3,
            face_color=(1.0, 1.0, 1.0, 0.9),
            edge_width=0,
        )

    # ==========================================================
    # camera - bridges ui_ux.py's CameraController onto vispy
    # ==========================================================

    def apply_camera_controller(self, camera_controller):
        """
        Pushes CameraController's logical (radius, theta, phi) state onto
        VisPy's own TurntableCamera. TurntableCamera expects
        azimuth/elevation in DEGREES, not radians.

        Call this only when camera_controller.detect_input() actually
        returned an action (i.e. a key is currently held), not
        unconditionally every frame - writing to cam.azimuth/elevation/
        distance every single frame would fight with and effectively
        disable VisPy's own native mouse-drag orbiting, which is the
        exact behaviour visualisation.py.md says to keep relying on.
        """
        cam = self.view.camera
        cam.azimuth = float(np.degrees(camera_controller.theta))
        cam.elevation = float(np.degrees(camera_controller.phi))
        cam.distance = float(camera_controller.radius)

    # ==========================================================
    # update loop (60 fps)
    # ==========================================================

    def update(self, event):
        """
        PERFORMANCE FIX: this used to call surface.set_data()/level=...
        and _update_tracers() unconditionally, every tick, regardless of
        whether sim_state had actually changed. Marching cubes on the
        default 240x240x400 grid (23M cells) is expensive - redoing it
        60 times a second while sitting at sim_status="init" (where
        solvers.py never even runs) meant the app was doing that full
        expensive rebuild forever at idle, for zero visible benefit,
        which is exactly what "slow even without starting anything" was.

        Now only rebuilds when something that actually affects the
        surface has changed: the physics has advanced a step (n_dt
        changed) or the iso-level slider itself moved. At true idle -
        sim paused/not started, slider untouched - this does no
        marching-cubes work at all after the first frame.
        """
        self.iso_alpha = self.slider.value

        data_changed = self.sim_state.n_dt != self._last_rendered_n_dt
        level_changed = self.iso_alpha != self._last_rendered_alpha

        if data_changed or level_changed:
            self.scalar_field = self.sim_state.rho
            self.surface.set_data(self.scalar_field)
            self.surface.level = self._compute_iso_value()
            self._last_rendered_n_dt = self.sim_state.n_dt
            self._last_rendered_alpha = self.iso_alpha

        if data_changed:
            self._update_tracers()

    def mark_dirty(self):
        """
        Forces the next update() tick to rebuild the isosurface/tracers
        regardless of n_dt. Needed after main.py swaps in a fresh
        SimState on RESTART: the new sim_state starts at n_dt=0, which
        could coincidentally equal whatever n_dt was last rendered before
        the restart (e.g. if the sim was restarted before ever taking a
        step) - without this, the dirty-check in update() would wrongly
        conclude nothing changed and leave the old isosurface on screen.
        """
        self._last_rendered_n_dt = -1

    # ==========================================================
    # slider
    # ==========================================================

    def _create_slider(self):
        """
        `from vispy.scene.widgets import Slider` (the original reference)
        throws ImportError on a real run - vispy.scene.widgets has never
        had a Slider class; that was very likely a GPT hallucination
        extrapolating a matplotlib-style widget VisPy doesn't provide.
        Rebuilt here as a self-contained IsoLevelSlider using the same
        Rectangle+Ellipse primitives ui_ux.py's ParameterSlider already
        uses successfully - not importing ParameterSlider itself, since
        it's built around a parameters.py Parameter object and iso_alpha
        isn't one. main.py's _ui_tick needs to call
        visualiser.slider.interact_mouse(mouse_pos, mouse_down) and
        .render() each frame, the same way it already does for every
        other slider.
        """
        slider_width, slider_height = 400, 40
        canvas_width, _ = self.canvas.size

        # centred horizontally, near the TOP of the screen. The original
        # reference hardcoded y=740 on an 800-tall canvas - in vispy
        # widget-space (y=0 at top) that sits near the bottom, not the
        # top as the spec asked for.
        slider_x = (canvas_width - slider_width) / 2.0
        slider_y = 20

        self.slider = IsoLevelSlider(
            parent=self.canvas.scene,
            position=(slider_x, slider_y),
            value=self.iso_alpha,
            min_value=0.05,
            max_value=0.95,
            length=slider_width,
            height=slider_height,
        )


class IsoLevelSlider:
    """
    Minimal drag slider for a single plain float (iso_alpha), not tied to
    a parameters.py Parameter object. See _create_slider's docstring for
    why this exists instead of vispy.scene.widgets.Slider (which doesn't
    exist) or ui_ux.py's ParameterSlider (which needs a real Parameter).
    """

    def __init__(self, parent, position, value, min_value, max_value, length, height):
        self.parent = parent
        self.position = position
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.length = length
        self.height = height

        self._dragging = False
        self._track_visual = None
        self._knob_visual = None

    def _value_to_knob_x(self):
        frac = (self.value - self.min_value) / (self.max_value - self.min_value)
        x, _ = self.position
        return x + frac * self.length

    def _knob_x_to_value(self, knob_x):
        x, _ = self.position
        frac = (knob_x - x) / self.length
        frac = min(max(frac, 0.0), 1.0)
        return self.min_value + frac * (self.max_value - self.min_value)

    def interact_mouse(self, mouse_pos, is_mouse_down):
        mx, my = mouse_pos
        knob_x = self._value_to_knob_x()
        _, y = self.position

        near_knob = (
            abs(mx - knob_x) <= self.height / 2.0
            and abs(my - (y + self.height / 2.0)) <= self.height / 2.0
        )

        if is_mouse_down and (self._dragging or near_knob):
            self._dragging = True
            self.value = self._knob_x_to_value(mx)
        else:
            self._dragging = False

    def render(self):
        x, y = self.position
        track_center = (x + self.length / 2.0, y + self.height / 2.0)
        knob_x = self._value_to_knob_x()
        knob_center = (knob_x, y + self.height / 2.0)

        if self._track_visual is None:
            self._track_visual = visuals.Rectangle(
                center=track_center,
                width=self.length,
                height=self.height,
                color="black",
                border_color="white",
                border_width=2,
                parent=self.parent,
            )
            self._knob_visual = visuals.Ellipse(
                center=knob_center,
                radius=self.height / 2.0,
                color="white",
                parent=self.parent,
            )
        else:
            self._knob_visual.center = knob_center