"""
main.py

Wires every module together into the runnable program. This is where the
five open wiring decisions from ui_ux.py / visualisation.py get resolved
concretely:

  1) Input capture: VisPy owns it entirely (mouse/keyboard events off the
     canvas). PyQt5 is only touched for the native export save dialog.
  2) Physics stepping and rendering run on two INDEPENDENT vispy Timers
     in the same single-threaded cooperative loop, not one-step-per-frame -
     the CFL-adaptive dt makes 1:1 coupling a bad fit (see chat).
  3) apply_camera_controller() is only called when a camera-relevant key
     is actually held, so it doesn't fight VisPy's native mouse orbiting.
  4) Export uses QFileDialog.getSaveFileName, not a browser "download".
  5) sim_status transitions are handled explicitly, once, at the moment
     a button click changes them - not re-evaluated from scratch every
     frame.

CAVEAT: unlike the previous eight files, this one is live GUI/event-loop
integration across PyQt5 and VisPy that could not be run or clicked
through in the environment this was written in. Treat this as a strong
first draft to actually run - some integration details (especially
canvas.scene layering for the 2D overlay widgets, discussed inline below)
are the kind of thing that's fast to confirm by running and slow to be
fully certain about from the spec alone.
"""

import vispy
vispy.use(app="pyqt5")
# Forcing the PyQt5 backend explicitly rather than letting vispy
# auto-select. PyQt5 is in requirements.txt, but vispy doesn't guarantee
# it'll pick that backend just because it's installed - and the export
# button below needs a real, single Qt event loop coexisting with vispy's
# loop, not two different toolkits' loops running side by side.

from vispy import app
from PyQt5 import QtWidgets

from parameters import InitParams, RTParams
from simstate import SimState
from solvers import evolve_simstate
from ui_ux import (
    CameraController,
    RuntimeDataDisplay,
    Button,
    InitSlider,
    RTSlider,
    TutorialBox,
    ParameterSlider,
)
from visualisation import ZPinchVisualiser
from exports_diagnostics import export_diagnostics


class Application:
    """
    Owns all the mutable, shared state that main.py's various callbacks
    (button clicks, the two timers, mouse/key events) all need to read
    and write - sim_state gets replaced wholesale on RESTART, sliders
    need to reflect that, timers need to check sim_status, etc. A plain
    while-loop (as main.py.md's pseudocode sketches) doesn't have a
    natural home for that shared, mutable state across many independent
    callbacks; a class does. This is the one deliberate departure from
    the flat-function style used everywhere else in the project - main.py
    is an orchestration layer, not physics/data logic, and needed it.
    """

    def __init__(self):
        # a QApplication must exist before any QFileDialog call later;
        # vispy's pyqt5 backend normally creates one when the canvas is
        # made, but this guards against relying on that implicitly.
        self.qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        self.init_params = InitParams()
        self.rt_params = RTParams()
        self.sim_state = SimState(self.init_params, self.rt_params)

        self.visualiser = ZPinchVisualiser(self.sim_state)
        self.canvas = self.visualiser.canvas

        self.camera_controller = CameraController()

        self.input_state = {
            "mouse_pos": (0, 0),
            "mouse_down": False,
            "keys_held": set(),
        }

        self._build_ui()
        self._wire_events()

        # two independent timers, not one physics-step-per-render-frame -
        # solvers.py's dt is CFL-adaptive, so locking it to a fixed 60Hz
        # render cadence would make a violent collapse (small dt) crawl
        # and a quiet phase (large dt) race ahead for no physical reason.
        self.ui_timer = app.Timer(interval=1 / 60, connect=self._ui_tick, start=True)

        # not literally 0 - a zero-interval timer in this single-threaded
        # cooperative loop could starve the UI timer and input events
        # entirely by never yielding. 1ms still runs physics far faster
        # than 60Hz whenever the CFL dt allows it, without locking out
        # everything else.
        self.physics_timer = app.Timer(interval=0.001, connect=self._physics_tick, start=True)

    # =====================================================
    # ui construction
    # =====================================================

    def _build_ui(self):
        canvas_w, canvas_h = self.canvas.size

        # ---- parameter sliders ----
        # NOTE: everything below is parented to self.canvas.scene, the
        # same fixed/screen-space layer visualisation.py already uses for
        # its iso-level Slider - as opposed to self.visualiser.view.scene,
        # which is the 3D scene the TurntableCamera rotates. This is the
        # biggest unverified assumption in this file (see module
        # docstring) - if sliders/buttons/text show up rotating with the
        # camera instead of staying fixed on screen, this is where to
        # look first.
        left_x = 40
        right_x = canvas_w - 40 - ParameterSlider.LENGTH
        start_y = 150
        spacing = 60

        init_specs = [
            ("n_particles", self.init_params.n_particles),
            ("cylinder_radius", self.init_params.cylinder_radius),
            ("cylinder_length", self.init_params.cylinder_length),
            ("e_coeff", self.init_params.e_coeff),
        ]
        rt_specs = [
            ("axial_current", self.rt_params.axial_current),
            ("temp_ext", self.rt_params.temp_ext),
            ("pressure_ext", self.rt_params.pressure_ext),
            ("grav", self.rt_params.grav),
            ("resistivity", self.rt_params.resistivity),
            ("viscosity", self.rt_params.viscosity),
        ]

        self.init_sliders = []
        for i, (name, parameter) in enumerate(init_specs):
            pos = (left_x, start_y + i * spacing)
            slider = InitSlider(parameter, pos, name)
            slider.parent = self.canvas.scene
            self.init_sliders.append(slider)

        self.rt_sliders = []
        for i, (name, parameter) in enumerate(rt_specs):
            pos = (right_x, start_y + i * spacing)
            slider = RTSlider(parameter, pos, name)
            slider.parent = self.canvas.scene
            self.rt_sliders.append(slider)

        # ---- tutorial box (bottom-right, init only) ----
        tutorial_pos = (canvas_w - 440, canvas_h - 440)
        self.tutorial_box = TutorialBox(position=tutorial_pos, parent=self.canvas.scene)

        # ---- runtime data display (top-left, while running/paused) ----
        self.runtime_display = RuntimeDataDisplay(parent=self.canvas.scene)

        # ---- buttons ----
        bottom_y = canvas_h - 90

        self.main_button = Button("START", (0.2, 0.6, 0.2), (40, bottom_y), parent=self.canvas.scene)
        self.main_button.display = True

        self.restart_button = Button(
            "RESTART", (0.5, 0.5, 0.5), (160, bottom_y), parent=self.canvas.scene
        )
        self.restart_default_button = Button(
            "RESTART DEFAULT", (0.5, 0.3, 0.3), (280, bottom_y), parent=self.canvas.scene
        )

        # export button occupies the same footprint the tutorial box used
        # (per spec: "replacing where the tutorial was"), centred within
        # it since it's smaller (150x150 vs the tutorial's 400x400).
        tut_x, tut_y = tutorial_pos
        export_center = (tut_x + 200, tut_y + 200)
        export_pos = (export_center[0] - 75, export_center[1] - 75)
        self.export_button = Button(
            "EXPORT DATA AND DIAGNOSTICS",
            (0.2, 0.4, 0.8),
            export_pos,
            length=150,
            height=150,
            parent=self.canvas.scene,
        )

    # =====================================================
    # input wiring - vispy owns all of it (see module docstring, #1)
    # =====================================================

    def _wire_events(self):
        @self.canvas.events.mouse_move.connect
        def on_mouse_move(event):
            # sliced to 2 elements defensively - Button.contains and
            # ParameterSlider.interact_mouse both unpack this as exactly
            # (mx, my), and event.pos isn't guaranteed to always be a
            # bare 2-tuple across every vispy backend/version.
            self.input_state["mouse_pos"] = tuple(event.pos[:2])

        @self.canvas.events.mouse_press.connect
        def on_mouse_press(event):
            self.input_state["mouse_down"] = True

        @self.canvas.events.mouse_release.connect
        def on_mouse_release(event):
            self.input_state["mouse_down"] = False

        @self.canvas.events.key_press.connect
        def on_key_press(event):
            if event.key is not None:
                self.input_state["keys_held"].add(event.key.name)

        @self.canvas.events.key_release.connect
        def on_key_release(event):
            if event.key is not None:
                self.input_state["keys_held"].discard(event.key.name)

    # =====================================================
    # per-frame ui update (60hz)
    # =====================================================

    def _ui_tick(self, event):
        mouse_pos = self.input_state["mouse_pos"]
        mouse_down = self.input_state["mouse_down"]
        keys_held = self.input_state["keys_held"]

        # camera - only touch it on an actual key press (see #3): calling
        # apply_camera_controller unconditionally every frame would fight
        # with and disable vispy's native mouse-drag orbiting.
        action = self.camera_controller.detect_input(keys_held)
        if action:
            self.camera_controller.change_camera(action)
            self.visualiser.apply_camera_controller(self.camera_controller)

        # sliders
        for slider in self.init_sliders + self.rt_sliders:
            slider.interact_mouse(mouse_pos, mouse_down)
            slider.render_slider()

        # buttons
        if self.main_button.detect_interaction(mouse_pos, mouse_down):
            self._on_main_button_click()
        if self.restart_button.detect_interaction(mouse_pos, mouse_down):
            self._on_restart_click(keep_current_values=True)
        if self.restart_default_button.detect_interaction(mouse_pos, mouse_down):
            self._on_restart_click(keep_current_values=False)
        if self.export_button.detect_interaction(mouse_pos, mouse_down):
            self._on_export_click()

        self.main_button.render()
        self.restart_button.render()
        self.restart_default_button.render()
        self.export_button.render()

        # visibility driven by sim_status (see #5)
        status = self.sim_state.sim_status

        for slider in self.init_sliders:
            slider.show_bool = status == "init"
        self.tutorial_box.render(visible=(status == "init"))

        if status in ("run", "pause"):
            self.runtime_display.update_from_simstate(self.sim_state)
        else:
            self.runtime_display.clear()

    # =====================================================
    # per-tick physics update (as fast as stable, see #2)
    # =====================================================

    def _physics_tick(self, event):
        if self.sim_state.sim_status == "run":
            evolve_simstate(self.sim_state)

    # =====================================================
    # button handlers - sim_status transitions (see #5)
    # =====================================================

    def _on_main_button_click(self):
        status = self.sim_state.sim_status

        if status == "init":
            self.sim_state.sim_status = "run"
            self.main_button.text = "PAUSE"
            self.restart_button.display = True
            self.restart_default_button.display = True

        elif status == "run":
            self.sim_state.sim_status = "pause"
            self.main_button.text = "RESUME"
            self.export_button.display = True

        elif status == "pause":
            self.sim_state.sim_status = "run"
            self.main_button.text = "PAUSE"
            self.export_button.display = False

    def _on_restart_click(self, keep_current_values):
        """
        RESTART keeps the current slider-adjusted parameter values and
        just re-seeds a fresh SimState from them. RESTART DEFAULT resets
        every parameter back to its own default first, THEN re-seeds -
        the closest thing to "restart the whole program" without an
        actual process relaunch. Both return to sim_status "init", which
        naturally brings the init sliders/tutorial/START button back and
        hides the restart buttons again (same UI state as first launch).
        """
        if not keep_current_values:
            for parameter in self._all_parameters():
                parameter.set_default()

        self.sim_state = SimState(self.init_params, self.rt_params)
        self.visualiser.sim_state = self.sim_state

        for slider in self.init_sliders + self.rt_sliders:
            slider.value = slider.parameter.value_rn

        self.sim_state.sim_status = "init"
        self.main_button.text = "START"
        self.restart_button.display = False
        self.restart_default_button.display = False
        self.export_button.display = False

    def _all_parameters(self):
        return [
            self.init_params.n_cells_x,
            self.init_params.n_cells_y,
            self.init_params.n_cells_z,
            self.init_params.cell_length,
            self.init_params.n_particles,
            self.init_params.cylinder_radius,
            self.init_params.cylinder_length,
            self.init_params.e_coeff,
            self.rt_params.axial_current,
            self.rt_params.temp_ext,
            self.rt_params.pressure_ext,
            self.rt_params.grav,
            self.rt_params.resistivity,
            self.rt_params.viscosity,
        ]

    def _on_export_click(self):
        """Native save dialog, not a browser download (see #4)."""
        if self.sim_state.sim_status != "pause":
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Export Diagnostics",
            "diagnostics_export.zip",
            "Zip files (*.zip)",
        )
        if path:
            export_diagnostics(self.sim_state, output_path=path)

    # =====================================================
    # entry point
    # =====================================================

    def run(self):
        app.run()


if __name__ == "__main__":
    Application().run()